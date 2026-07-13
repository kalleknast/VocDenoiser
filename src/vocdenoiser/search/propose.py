"""Candidate proposers — the search operators.

The loop asks a proposer for the next candidate given the ledger so far. Three
strategies compose:

- **bootstrap**: sample random candidates until there is a frontier to exploit;
- **evolve**: mutate a frontier member or cross two of them (FunSearch-style);
- **LLM** (optional): hand the policy (``program.md``) + the frontier + recent
  history to an injected ``llm_fn`` that returns the next candidate as JSON.
  Falls back to ``evolve`` if no ``llm_fn`` is supplied or its output is invalid.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from vocdenoiser.search.ledger import Ledger
from vocdenoiser.search.space import Candidate, crossover, mutate, random_candidate

LLMFn = Callable[[str], str]  # prompt -> JSON candidate (possibly with extra keys)

_PROGRAM = Path(__file__).with_name("program.md")


class Proposer:
    def __init__(
        self,
        init_random: int = 6,
        frontier_k: int = 8,
        p_crossover: float = 0.3,
        explore_rate: float = 0.0,
        llm_fn: LLMFn | None = None,
        llm_every: int = 1,
    ) -> None:
        self.init_random = init_random
        self.frontier_k = frontier_k
        self.p_crossover = p_crossover
        # ε-greedy exploration: even once a frontier exists, propose a fresh random
        # candidate this fraction of the time. Pure hill-climbing off a frontier that
        # converged under a small budget stays stuck in one basin (the ledger showed
        # a plateau over iters 30-58); random restarts re-seed distant regions.
        self.explore_rate = explore_rate
        self.llm_fn = llm_fn
        self.llm_every = llm_every
        self._n = 0
        self.last_rationale = ""

    def propose(self, ledger: Ledger, rng: np.random.RandomState) -> Candidate:
        self._n += 1
        self.last_rationale = ""
        records = ledger.load()
        frontier = ledger.frontier(self.frontier_k)

        if len(records) < self.init_random or not frontier:
            return random_candidate(rng, origin="random")

        if self.explore_rate and rng.random() < self.explore_rate:
            self.last_rationale = "explore (ε-greedy random restart)"
            return random_candidate(rng, origin="random")

        if self.llm_fn is not None and (self._n % self.llm_every == 0):
            cand = self._propose_llm(ledger, frontier)
            if cand is not None:
                return cand

        return self._evolve(frontier, rng)

    def _evolve(self, frontier, rng) -> Candidate:
        cands = [Candidate.from_dict(r.candidate) for r in frontier]
        if len(cands) >= 2 and rng.random() < self.p_crossover:
            a, b = rng.choice(len(cands), size=2, replace=False)
            return crossover(cands[a], cands[b], rng)
        # Bias mutation toward the better frontier members.
        weights = np.linspace(len(cands), 1, len(cands))
        weights = weights / weights.sum()
        parent = cands[int(rng.choice(len(cands), p=weights))]
        return mutate(parent, rng, n_edits=int(rng.choice([1, 1, 2])))

    def _propose_llm(self, ledger: Ledger, frontier) -> Candidate | None:
        prompt = build_llm_prompt(ledger, frontier)
        try:
            raw = self.llm_fn(prompt)
            data = json.loads(raw)
            self.last_rationale = data.get("rationale", "")
            cand = Candidate.from_dict(data.get("candidate", data))
            cand = Candidate(**{**cand.to_dict(), "origin": "llm"})
            cand.validate()
            return cand
        except Exception:  # noqa: BLE001 - invalid LLM output -> fall back to evolve
            return None


def build_llm_prompt(ledger: Ledger, frontier, recent: int = 12) -> str:
    """Assemble the proposer prompt: policy + frontier + recent history.

    Mirrors autoresearch's context economy — feed the distilled ledger, not raw
    logs — so the model reasons over what's been tried without drowning in it.
    """
    policy = _PROGRAM.read_text() if _PROGRAM.exists() else "(policy file missing)"
    fro = "\n".join(
        f"  {r.metric:+.3f}±{r.metric_std:.2f} [{r.num_params/1e6:.2f}M] {r.candidate}"
        for r in frontier
    )
    hist = "\n".join(
        f"  {r.status:7s} {r.metric:+.3f} {r.origin:9s} {r.description}"
        for r in ledger.load()[-recent:]
    )
    schema = _candidate_schema()
    return (
        f"{policy}\n\n"
        f"## Current frontier (best kept candidates, metric higher=better)\n{fro}\n\n"
        f"## Recent experiments\n{hist}\n\n"
        f"## Candidate JSON schema (emit exactly these keys)\n{schema}\n\n"
        "Respond with a single JSON object: "
        '{\"rationale\": \"<one line hypothesis>\", \"candidate\": { ...knobs... }}. '
        "Propose ONE new candidate that is a small, motivated edit of a strong "
        "frontier member (or a crossover), not a repeat of anything already tried."
    )


def _candidate_schema() -> str:
    from vocdenoiser.search.space import CHOICES, LOG_RANGES

    lines = [f"  {k}: one of {v}" for k, v in CHOICES.items()]
    lines += [f"  {k}: float in {v}" for k, v in LOG_RANGES.items()]
    return "\n".join(lines)
