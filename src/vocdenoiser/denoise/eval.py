"""Evaluate the learned 16-dim latent space (SPECS.md evaluation metrics).

  * **Latent separability** — UMAP of the 16 VAE features to a 2-D scatter,
    coloured by individual identity, saved as a PNG.
  * **Identity-classification proxy** — a RandomForest trained on the 16 features
    with stratified cross-validation; accuracy well above chance means individual
    identity survives the compression.

Latents are the encoder means ``mu`` of *clean* calls (augmentation off), i.e.
the deterministic embedding used downstream for clustering.

    python -m vocdenoiser.denoise.eval --ckpt checkpoints/last.ckpt \
        --labels-csv individuals.csv

Individual-identity labels for the RandomForest proxy come from one of:
  * ``--labels-csv`` — a CSV mapping call ID → individual (columns default to
    ``id,identity``; the ID is matched against each WAV's stem or filename). This
    is the path for ``data/Vocalizations``, whose files are bare numeric call IDs
    with no identity in the filename.
  * ``--label-from`` — derive from the path when identity *is* encoded there:
    ``parent`` (folder), ``stem`` (filename), or ``prefix`` (text before the first
    ``--label-sep``, default ``_``).

With no labels, the UMAP scatter still renders; the RandomForest proxy is skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vocdenoiser.denoise.config import Config


def _label_for(path: Path, scheme: str, sep: str) -> str:
    if scheme == "parent":
        return path.parent.name
    if scheme == "stem":
        return path.stem
    if scheme == "prefix":
        return path.stem.split(sep)[0]
    raise ValueError(f"Unknown --label-from scheme: {scheme!r}")


def _load_label_map(csv_path: str, key_col: str, val_col: str) -> dict[str, str]:
    """Read a CSV into a ``{call-id: identity}`` dict for the RF identity proxy."""
    import csv

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or key_col not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} needs a header with a '{key_col}' column "
                f"(got {reader.fieldnames}). Set --labels-key-col / --labels-value-col."
            )
        return {row[key_col]: row[val_col] for row in reader}


def _resolve_labels(files: list[Path], args) -> list[str | None]:
    """Per-file identity label (None where unknown), from CSV or the path scheme."""
    if args.labels_csv:
        lmap = _load_label_map(args.labels_csv, args.labels_key_col, args.labels_value_col)
        labels = [lmap.get(p.stem, lmap.get(p.name)) for p in files]
        n_missing = sum(label is None for label in labels)
        if n_missing:
            print(f"{n_missing}/{len(files)} calls had no entry in {args.labels_csv}.")
        return labels
    return [_label_for(p, args.label_from, args.label_sep) for p in files]


def extract_latents(cfg: Config, ckpt_path: str) -> tuple[np.ndarray, list[Path]]:
    """Encode every clean call to its 16-dim latent mean ``mu``."""
    import torch
    from torch.utils.data import DataLoader

    from vocdenoiser.denoise.beta_vae import BetaVAE
    from vocdenoiser.denoise.dataset import PheeDenoiseDataset, list_clean_calls

    eval_cfg = Config(**{**cfg.__dict__, "augment": False})
    files = list_clean_calls(eval_cfg)
    ds = PheeDenoiseDataset(eval_cfg, files)
    dl = DataLoader(ds, batch_size=eval_cfg.batch_size, num_workers=eval_cfg.num_workers)

    model = BetaVAE.load_from_checkpoint(ckpt_path, cfg=eval_cfg)
    model.eval()
    device = model.device

    latents: list[np.ndarray] = []
    with torch.no_grad():
        for clean_spec, _ in dl:
            mu, _ = model.encode(clean_spec.to(device))
            latents.append(mu.cpu().numpy())
    return np.concatenate(latents, axis=0), files


def run_umap(latents: np.ndarray, labels: list[str], out_png: str, seed: int) -> None:
    """UMAP the latents to 2-D and save a scatter coloured by identity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    emb = umap.UMAP(n_components=2, random_state=seed).fit_transform(latents)
    uniq = sorted(set(labels))
    idx = {u: i for i, u in enumerate(uniq)}
    colors = np.array([idx[label] for label in labels])

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=colors, cmap="tab20", s=8, alpha=0.8)
    ax.set(title="UMAP of 16-dim β-VAE latents", xlabel="UMAP-1", ylabel="UMAP-2")
    if len(uniq) <= 20:
        ax.legend(*scatter.legend_elements(), title="identity", loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved UMAP scatter -> {out_png}")


def identity_rf(latents: np.ndarray, labels: list[str], seed: int) -> float:
    """Cross-validated RandomForest identity accuracy on the 16 features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    y = np.array(labels)
    _, counts = np.unique(y, return_counts=True)
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        print("Too few samples per identity for cross-validation; skipping RF proxy.")
        return float("nan")

    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, latents, y, cv=cv, scoring="accuracy")
    chance = counts.max() / counts.sum()
    print(
        f"RandomForest identity accuracy: {scores.mean():.3f} ± {scores.std():.3f} "
        f"(chance ≈ {chance:.3f}, {len(np.unique(y))} identities, cv={n_splits})"
    )
    return float(scores.mean())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate β-VAE latents.")
    Config.add_cli_args(parser)
    parser.add_argument("--ckpt", required=True, help="path to a trained .ckpt")
    parser.add_argument("--labels-csv", help="CSV mapping call id -> individual identity")
    parser.add_argument("--labels-key-col", default="id", help="CSV id column (matches WAV stem)")
    parser.add_argument("--labels-value-col", default="identity", help="CSV identity column")
    parser.add_argument("--label-from", default="parent", choices=["parent", "stem", "prefix"])
    parser.add_argument("--label-sep", default="_")
    parser.add_argument("--out-png", default="umap_latents.png")
    parser.add_argument("--out-latents", default="latents.npy")
    args = parser.parse_args(argv)
    cfg = Config.from_args(args)

    latents, files = extract_latents(cfg, args.ckpt)
    np.save(args.out_latents, latents)
    print(f"Extracted {latents.shape[0]} latents of dim {latents.shape[1]} -> {args.out_latents}")

    labels = _resolve_labels(files, args)
    run_umap(latents, [label or "unlabeled" for label in labels], args.out_png, cfg.seed)

    keep = [i for i, label in enumerate(labels) if label is not None]
    if len(keep) >= 2 and len({labels[i] for i in keep}) >= 2:
        identity_rf(latents[keep], [labels[i] for i in keep], cfg.seed)
    else:
        print(
            "No usable identity labels — skipping the RandomForest proxy. "
            "Pass --labels-csv (id,identity) to enable it."
        )


if __name__ == "__main__":
    main()
