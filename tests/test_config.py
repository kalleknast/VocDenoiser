"""Config CLI plumbing, incl. the real-noise fields (torch-free)."""

import argparse

from vocdenoiser.denoise.config import Config


def _parse(argv):
    p = argparse.ArgumentParser()
    Config.add_cli_args(p)
    return Config.from_args(p.parse_args(argv))


def test_noise_dirs_defaults():
    cfg = Config()
    assert cfg.noise_dirs == ("data/Noise", "data/Cigarra")
    assert cfg.real_noise_weight == 0.5


def test_noise_dirs_list_cli():
    cfg = _parse(["--noise-dirs", "/content/Noise", "/content/Cigarra"])
    assert cfg.noise_dirs == ("/content/Noise", "/content/Cigarra")
    assert [str(d) for d in cfg.resolved_noise_dirs()] == ["/content/Noise", "/content/Cigarra"]


def test_real_noise_weight_cli():
    cfg = _parse(["--real-noise-weight", "0.25"])
    assert cfg.real_noise_weight == 0.25


def test_scalar_and_bool_cli_still_work():
    cfg = _parse(["--beta", "2.0", "--no-augment"])
    assert cfg.beta == 2.0
    assert cfg.augment is False
