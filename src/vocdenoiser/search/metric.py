"""The frozen search metric: spectrogram-domain SI-SDR (dB, higher = better).

Scale-invariant SDR between the denoised spectrogram and the clean-target
spectrogram, averaged over a pinned validation set. Scale invariance makes it
robust to overall level differences between candidates, and computing it in the
fixed spectrogram domain (the harness pins the geometry) keeps it comparable
across candidates that differ only in their model. Requires torch.
"""

from __future__ import annotations

import torch


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR (dB) per sample; ``estimate``/``reference`` are (B, ...) tensors."""
    est = estimate.flatten(1)
    ref = reference.flatten(1)
    ref = ref - ref.mean(dim=1, keepdim=True)
    est = est - est.mean(dim=1, keepdim=True)
    alpha = (est * ref).sum(dim=1, keepdim=True) / (ref.pow(2).sum(dim=1, keepdim=True) + eps)
    target = alpha * ref
    noise = est - target
    ratio = (target.pow(2).sum(dim=1) + eps) / (noise.pow(2).sum(dim=1) + eps)
    return 10.0 * torch.log10(ratio)


@torch.no_grad()
def spectrogram_si_sdr(model, val_dl, max_batches: int = 20) -> float:
    """Mean SI-SDR (dB) of the model's reconstruction over ``val_dl``."""
    model.eval()
    device = next(model.parameters()).device
    vals = []
    for i, (noisy, clean) in enumerate(val_dl):
        if i >= max_batches:
            break
        noisy, clean = noisy.to(device), clean.to(device)
        out = model(noisy)
        recon = out[0] if isinstance(out, (tuple, list)) else out
        vals.append(si_sdr(recon, clean).cpu())
    if not vals:
        return float("nan")
    return float(torch.cat(vals).mean())
