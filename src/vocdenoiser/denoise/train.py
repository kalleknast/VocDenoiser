"""Train the β-VAE denoiser with PyTorch Lightning.

Environment-agnostic: the clean-call root comes from ``--data-root`` or the
``VOCDENOISER_DATA_ROOT`` env var (never hardcoded), so the same command runs on
Colab and on a local box. Every :class:`~vocdenoiser.denoise.config.Config` field
is exposed as a CLI flag.

    python -m vocdenoiser.denoise.train --data-root /path/to/clean_calls \
        --max-epochs 100 --batch-size 32 --beta 4.0

Copy the clean set to local disk first — do not train off the pCloud mount.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from vocdenoiser.denoise.config import Config


def _best_ckpt(ckpt_dir: Path, stem: str = "betavae") -> Path | None:
    """Lowest-val_loss finite checkpoint in ``ckpt_dir`` (skips NaN-named files)."""
    import re

    cands = [
        p for p in ckpt_dir.glob(f"{stem}-*val_loss=*.ckpt") if "nan" not in p.name.lower()
    ]
    if not cands:
        return None
    return min(
        cands, key=lambda p: float(re.search(r"val_loss=([0-9]+\.[0-9]+)", p.name).group(1))
    )


def _resolve_resume(resume_from: str | None, cfg: Config, stem: str = "betavae") -> str | None:
    """Map ``--resume-from`` to a ``trainer.fit(ckpt_path=...)`` value (None = fresh)."""
    if not resume_from:
        return None
    ckpt_dir = cfg.resolved_ckpt_dir()
    if resume_from in ("auto", "last"):
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            print(f"Resuming full training state from {last}")
            return str(last)
        print(f"--resume-from {resume_from}: no last.ckpt in {ckpt_dir} — starting fresh.")
        return None
    if resume_from == "best":
        best = _best_ckpt(ckpt_dir, stem)
        if best is not None:
            print(f"Resuming from best checkpoint {best}")
            return str(best)
        print(f"--resume-from best: no finite checkpoint in {ckpt_dir} — starting fresh.")
        return None
    print(f"Resuming from {resume_from}")
    return resume_from


def candidate_from_ledger(ledger_path: str, candidate_id: str):
    """Look up one search candidate by id (or ``"best"``) in a search ledger.

    The bridge from a search result to a trainable model: the ledger is the ONLY record of
    what the search scored — the harness trains each candidate for a fixed budget with
    checkpointing off and throws the weights away.
    """
    from vocdenoiser.search.ledger import Ledger
    from vocdenoiser.search.space import Candidate

    ledger = Ledger(ledger_path)
    records = ledger.load()
    if not records:
        raise SystemExit(f"--from-ledger {ledger_path}: no records found")

    if candidate_id == "best":
        rec = ledger.best()
        if rec is None:
            raise SystemExit(f"--from-ledger {ledger_path}: no kept candidate to take as 'best'")
    else:
        rec = next((r for r in records if r.id == candidate_id), None)
        if rec is None:
            top = ", ".join(
                f"{r.id} ({r.metric:+.4f})"
                for r in sorted((x for x in records if x.status != "crash"),
                                key=lambda x: -x.metric)[:5]
            )
            raise SystemExit(
                f"--candidate-id {candidate_id!r} not in {ledger_path}.\n"
                f"Best available: {top}\nOr pass --candidate-id best."
            )
    if rec.status == "crash":
        raise SystemExit(
            f"candidate {rec.id} is recorded as a CRASH (metric -inf) — it never trained. "
            f"Pick another id."
        )
    print(
        f"Candidate {rec.id} from {ledger_path}: search metric {rec.metric:+.4f}"
        f"±{rec.metric_std:.3f}, {rec.num_params:,} params ({rec.status}, origin={rec.origin})"
    )
    return Candidate.from_dict(rec.candidate), rec


def build_dataloaders(cfg: Config):
    """Split discovered calls into train/val and wrap them in DataLoaders."""
    import lightning as L
    from torch.utils.data import DataLoader

    from vocdenoiser.denoise.dataset import PheeDenoiseDataset, list_clean_calls

    files = list_clean_calls(cfg)
    L.seed_everything(cfg.seed, workers=True)

    n_val = max(1, int(len(files) * cfg.val_frac))
    val_files, train_files = files[:n_val], files[n_val:]

    train_ds = PheeDenoiseDataset(cfg, train_files)
    val_ds = PheeDenoiseDataset(cfg, val_files)  # noise kept on in val too

    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,  # keep workers (+ babble cache) across epochs
    )
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_dl = DataLoader(val_ds, shuffle=False, **common)
    return train_ds, train_dl, val_dl


def main(argv: list[str] | None = None) -> None:
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

    from vocdenoiser.denoise.beta_vae import BetaVAE
    from vocdenoiser.search.model_factory import build_search_model

    # TF32 matmuls on Tensor-Core GPUs (L4/A100): a throughput win at negligible
    # precision cost for this task; harmless no-op on CPU/other GPUs.
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Train the β-VAE phee denoiser.")
    Config.add_cli_args(parser)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="resume training: 'auto'/'last' (last.ckpt if present, else fresh), "
        "'best' (lowest-val_loss ckpt), or an explicit .ckpt path. Checkpoints live "
        "under --output-root / $VOCDENOISER_OUTPUT_ROOT, so this resumes across resets.",
    )
    parser.add_argument(
        "--from-ledger",
        default=None,
        help="train an architecture-search WINNER instead of the hand-designed BetaVAE: "
        "path to a search ledger JSONL (pair with --candidate-id). The searched "
        "architecture (norm/act/depth/residual/... ) cannot be expressed as Config flags, "
        "so this is the only way to train what the search actually scored.",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="which candidate from --from-ledger to train; 'best' takes the ledger's "
        "current incumbent. Note the search metric is reconstruction SI-SDR only — it is "
        "blind to caller-identity retention, so 'best' is best-at-reconstruction.",
    )
    args = parser.parse_args(argv)
    cfg = Config.from_args(args)

    # A ledger candidate supersedes the model/optimisation flags: its architecture is what the
    # search scored, and its batch_size must drive the dataloaders (the search harness builds
    # Config the same way, so training matches scoring). Geometry/data flags stay from the CLI.
    # Checkpoints go to their own per-candidate subdir: several candidates get trained back to
    # back, they each write a `last.ckpt`, and mixing runs in one dir has bitten this project
    # before (a stale NaN `last.ckpt` poisoning a later eval).
    cand = None
    stem = "betavae"
    if args.from_ledger or args.candidate_id:
        if not (args.from_ledger and args.candidate_id):
            parser.error("--from-ledger and --candidate-id must be given together")
        cand, _rec = candidate_from_ledger(args.from_ledger, args.candidate_id)
        stem = f"search-{cand.id}"
        cfg = replace(
            cfg,
            **cand.to_config_overrides(),
            ckpt_dir=str(Path(cfg.ckpt_dir) / stem),
        )

    train_ds, train_dl, val_dl = build_dataloaders(cfg)
    model = build_search_model(cand, cfg) if cand is not None else BetaVAE(cfg)

    class _SetEpoch(Callback):
        def on_train_epoch_start(self, trainer, _pl_module):
            train_ds.set_epoch(trainer.current_epoch)

    ckpt = ModelCheckpoint(
        dirpath=str(cfg.resolved_ckpt_dir()),
        filename=stem + "-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    # Refresh the progress bar every 10 batches instead of every batch. In Colab's
    # non-TTY output each redraw is emitted as fresh lines rather than overwritten
    # in place, so a per-batch refresh floods the cell with thousands of lines;
    # refresh_rate=10 cuts that 10x. Prefer the Rich bar when `rich` is installed
    # (the nicer panel this run already renders), else the default tqdm bar.
    try:
        import rich  # noqa: F401

        from lightning.pytorch.callbacks import RichProgressBar

        progress_bar: Callback = RichProgressBar(refresh_rate=10)
    except ModuleNotFoundError:
        from lightning.pytorch.callbacks import TQDMProgressBar

        progress_bar = TQDMProgressBar(refresh_rate=10)

    callbacks: list[Callback] = [ckpt, _SetEpoch(), progress_bar]
    if cfg.early_stop_patience > 0:
        # Stop once val_loss stops improving; also halts on a non-finite val_loss
        # (check_finite defaults True), complementing the training-step NaN skip.
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=cfg.early_stop_patience,
                min_delta=cfg.early_stop_min_delta,
            )
        )
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # cap step size so a bad batch can't blow the VAE up to NaN
    )
    trainer.fit(model, train_dl, val_dl, ckpt_path=_resolve_resume(args.resume_from, cfg, stem))
    print(f"Best checkpoint: {ckpt.best_model_path}")


if __name__ == "__main__":
    main()
