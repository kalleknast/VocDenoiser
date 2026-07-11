"""Summarise a search ledger: running-best frontier, keep-rate, best candidate.

The analogue of autoresearch's ``analysis.ipynb`` — a compact, dependency-light
view of how the search progressed, suitable for the terminal or to hand back to
an LLM proposer.
"""

from __future__ import annotations

from pathlib import Path

from vocdenoiser.search.ledger import Ledger


def build_report(ledger_path: str | Path) -> str:
    ledger = Ledger(ledger_path)
    records = ledger.load()
    if not records:
        return "empty ledger"

    lines = ["# Architecture search report\n"]
    n = len(records)
    kept = [r for r in records if r.status == "keep"]
    crashed = [r for r in records if r.status == "crash"]
    lines.append(f"- experiments: **{n}**  (kept {len(kept)}, discarded "
                 f"{n - len(kept) - len(crashed)}, crashed {len(crashed)})")
    keep_rate = len(kept) / max(n - len(crashed), 1)
    lines.append(f"- keep-rate (of non-crashed): {100 * keep_rate:.0f}%\n")

    # running best (frontier over experiment index)
    lines.append("## running-best metric (higher = better)\n")
    lines.append("| experiment | origin | metric | status | running best |")
    lines.append("|---:|---|---:|:--:|---:|")
    best = float("-inf")
    for i, r in enumerate(records, 1):
        if r.status == "keep":
            best = max(best, r.metric)
        mark = {"keep": "✅", "discard": "·", "crash": "✗"}.get(r.status, "?")
        bstr = f"{best:+.3f}" if best != float("-inf") else "—"
        lines.append(f"| {i} | {r.origin} | {r.metric:+.3f} | {mark} | {bstr} |")
    lines.append("")

    b = ledger.best()
    if b:
        lines.append("## best candidate\n")
        lines.append(f"- metric **{b.metric:+.3f} ± {b.metric_std:.3f}** dB "
                     f"({b.num_params/1e6:.2f}M params, id `{b.id}`)")
        if b.llm_rationale:
            lines.append(f"- rationale: {b.llm_rationale}")
        lines.append("\n```json")
        import json

        lines.append(json.dumps(b.candidate, indent=2))
        lines.append("```")
    return "\n".join(lines)
