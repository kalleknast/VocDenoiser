"""VocDenoiser: denoising + compression of marmoset vocalizations.

Subpackages
-----------
- ``vocdenoiser.snr``    : call-agnostic SNR estimation and clean-subset selection (numpy-only).
- ``vocdenoiser.denoise``: conv / β-VAE spectrogram denoiser (PyTorch/Lightning; Colab/GPU).
- ``vocdenoiser.search`` : autoresearch-style architecture + hyperparameter search.
"""

__version__ = "0.1.0"
