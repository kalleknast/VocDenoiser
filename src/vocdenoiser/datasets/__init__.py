"""Prepare external labelled marmoset-call datasets for the VocDenoiser eval.

Each module turns a public dataset into (a) per-call WAV clips at a chosen sample
rate and (b) an ``id,identity`` CSV consumable by
``vocdenoiser.denoise.eval --labels-csv``:

* :mod:`vocdenoiser.datasets.infantmarmosetsvox` — caller identity + call type
  (Idiap; Zenodo 10130104, CC-BY-4.0). The open path for the identity proxy.
* :mod:`vocdenoiser.datasets.marmaudio` — call type only, 96 kHz
  (Lamothe et al., Sci Data 2025; Zenodo 15017207, CC-BY-4.0).
"""
