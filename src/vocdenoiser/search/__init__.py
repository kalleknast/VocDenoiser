"""Autoresearch-style architecture + hyperparameter search for the denoiser.

Inspired by Andrej Karpathy's ``autoresearch`` (LLM greedy hill-climb, fixed
compute budget as the equalizer, one scalar metric, append-only ledger, and a
two-level split between a *frozen harness*, an *editable candidate*, and a
human-tuned *policy* — see ``search/program.md``). It upgrades the toy-specific
parts for a noisier audio objective: a **typed candidate grammar** instead of
free-file edits, a **noise-aware accept** rule (metrics are seed-averaged and a
challenger must beat the incumbent by more than the noise band), and a **top-K
frontier with crossover** instead of a single greedy champion.

The loop, ledger, candidate grammar, accept rule and proposers are pure
numpy/stdlib and run (and are tested) without a GPU via :class:`MockHarness`.
Only the real :class:`TorchHarness` / model factory need torch + a GPU (Colab).
"""

from vocdenoiser.search.ledger import Ledger, Record
from vocdenoiser.search.space import Candidate

__all__ = ["Candidate", "Ledger", "Record"]
