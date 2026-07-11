"""Evaluate the learned 16-dim latent space (SPECS.md evaluation metrics).

  * **Latent separability** — UMAP of the 16 VAE features to a 2-D scatter,
    coloured by individual identity, saved as a PNG.
  * **Identity-classification proxy** — a RandomForest trained on the 16 features
    with stratified cross-validation; accuracy well above chance means individual
    identity survives the compression.

Latents are the encoder means ``mu`` of *clean* calls (augmentation off), i.e.
the deterministic embedding used downstream for clustering.

    python -m vocdenoiser.denoise.eval --data-root /path/to/clean_calls \
        --ckpt checkpoints/last.ckpt --label-from parent

Identity labels are derived from each file's path via ``--label-from``:
``parent`` (immediate folder name), ``stem`` (filename), or ``prefix`` (text
before the first ``--label-sep``, default ``_``).
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
    parser.add_argument("--label-from", default="parent", choices=["parent", "stem", "prefix"])
    parser.add_argument("--label-sep", default="_")
    parser.add_argument("--out-png", default="umap_latents.png")
    parser.add_argument("--out-latents", default="latents.npy")
    args = parser.parse_args(argv)
    cfg = Config.from_args(args)

    latents, files = extract_latents(cfg, args.ckpt)
    labels = [_label_for(p, args.label_from, args.label_sep) for p in files]
    np.save(args.out_latents, latents)
    print(f"Extracted {latents.shape[0]} latents of dim {latents.shape[1]} -> {args.out_latents}")

    run_umap(latents, labels, args.out_png, cfg.seed)
    identity_rf(latents, labels, cfg.seed)


if __name__ == "__main__":
    main()
