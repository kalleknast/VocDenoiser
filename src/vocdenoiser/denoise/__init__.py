"""Conv / β-VAE spectrogram denoiser (PyTorch/Lightning; Colab/GPU).

Supervised *noisy → clean* mapping for isolated marmoset phee calls, plus the
dynamic colony-noise augmentation that synthesises the training pairs on the fly
(see :mod:`vocdenoiser.denoise.augment` and ``SPECS.md``). The 16-dim latent is
the compressed feature used downstream for clustering / identity analysis.

The heavy ML stack (torch, torchaudio, lightning, umap, sklearn) lives in the
optional ``ml`` extra, so importing this subpackage requires ``pip install
-e '.[ml]'``. The numpy-only :mod:`vocdenoiser.snr` pipeline stays importable
without it.
"""

from vocdenoiser.denoise.config import Config

__all__ = ["Config"]
