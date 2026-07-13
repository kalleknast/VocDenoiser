"""The search loop: greedy hill-climb over a top-K frontier.

Each iteration: propose a candidate, skip it if already evaluated, run it under
the frozen harness across a few seeds, log the seed-averaged result, and update
the incumbent via the noise-aware accept rule. The frontier (kept in the ledger)
is the persisted state — resumable by pointing at the same JSONL file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vocdenoiser.search.accept import noise_aware_accept
from vocdenoiser.search.ledger import Ledger, Record
from vocdenoiser.search.propose import Proposer


@dataclass
class SearchConfig:
    iters: int = 40
    seeds: tuple[int, ...] = (0, 1)
    k_sigma: float = 1.0
    frontier_k: int = 8
    seed: int = 0
    max_skips: int = 50  # consecutive duplicate proposals before giving up
    # Prefer the smaller model when metrics are statistically tied. Turn off under
    # a small per-candidate budget, where compact models win only on train speed.
    simplicity_tiebreak: bool = True


def run_search(
    harness,
    ledger: Ledger,
    proposer: Proposer | None = None,
    cfg: SearchConfig | None = None,
    log=print,
) -> Record | None:
    """Run the search; returns the best :class:`Record`. ``harness`` implements
    ``evaluate(candidate, seeds) -> EvalResult``."""
    cfg = cfg or SearchConfig()
    proposer = proposer or Proposer(frontier_k=cfg.frontier_k)
    rng = np.random.RandomState(cfg.seed)
    seen = ledger.seen_ids()

    done = 0
    skips = 0
    while done < cfg.iters:
        cand = proposer.propose(ledger, rng)
        if cand.id in seen:
            skips += 1
            if skips > cfg.max_skips:
                log("proposer keeps repeating known candidates; stopping early")
                break
            continue
        skips = 0
        seen.add(cand.id)

        res = harness.evaluate(cand, list(cfg.seeds))
        incumbent = ledger.best()
        status = res.status
        decision_reason = ""
        if status == "keep" and incumbent is not None:
            dec = noise_aware_accept(
                res.metric, res.metric_std, incumbent.metric, incumbent.metric_std,
                k_sigma=cfg.k_sigma,
                challenger_params=res.num_params, incumbent_params=incumbent.num_params,
                simplicity_tiebreak=cfg.simplicity_tiebreak,
            )
            status = "keep" if dec.accept else "discard"
            decision_reason = dec.reason

        rec = Record(
            id=cand.id,
            candidate=cand.to_dict(),
            metric=res.metric,
            metric_std=res.metric_std,
            seeds=res.seeds,
            num_params=res.num_params,
            peak_vram_mb=res.peak_vram_mb,
            train_seconds=res.train_seconds,
            status=status,
            parent_id=cand.parent_id,
            origin=cand.origin,
            description=f"{cand.origin}: {decision_reason}".strip(": "),
            llm_rationale=proposer.last_rationale,
        )
        ledger.append(rec)
        done += 1
        best = ledger.best()
        # best is None until something is kept: a candidate that crashes/discards on
        # iteration 1 leaves the frontier empty, so guard the incumbent readout rather
        # than dereferencing None (which would abort the whole search on one bad candidate).
        best_str = f"{best.metric:+.3f} ({best.id})" if best is not None else "n/a"
        log(
            f"[{done:3d}/{cfg.iters}] {cand.origin:9s} metric={res.metric:+.3f}"
            f"±{res.metric_std:.2f} -> {status:7s}  best={best_str}"
        )

    return ledger.best()
