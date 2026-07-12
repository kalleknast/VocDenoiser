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

from vocdenoiser.denoise.config import Config


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
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

    from vocdenoiser.denoise.beta_vae import BetaVAE

    parser = argparse.ArgumentParser(description="Train the β-VAE phee denoiser.")
    Config.add_cli_args(parser)
    cfg = Config.from_args(parser.parse_args(argv))

    train_ds, train_dl, val_dl = build_dataloaders(cfg)
    model = BetaVAE(cfg)

    class _SetEpoch(Callback):
        def on_train_epoch_start(self, trainer, _pl_module):
            train_ds.set_epoch(trainer.current_epoch)

    ckpt = ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        filename="betavae-{epoch:02d}-{val_loss:.4f}",
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
    trainer.fit(model, train_dl, val_dl)
    print(f"Best checkpoint: {ckpt.best_model_path}")


if __name__ == "__main__":
    main()
