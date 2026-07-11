"""Append-only JSONL experiment ledger + the top-K frontier.

One JSON object per experiment: the full candidate, the metric (seed-averaged)
and its spread, resource costs, status, provenance, and the LLM's rationale.
JSONL (not TSV) so the nested candidate round-trips losslessly; append-only so a
crash never corrupts prior rows and the file is the reproducible record. Kept
compact enough to feed back into an LLM proposer's context.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vocdenoiser.search.space import Candidate


@dataclass
class Record:
    id: str
    candidate: dict
    metric: float  # PRIMARY scalar, higher is better (e.g. spectrogram SI-SDR, dB)
    metric_std: float = 0.0  # across seeds
    seeds: list[int] = field(default_factory=list)
    num_params: int = 0
    peak_vram_mb: float = 0.0
    train_seconds: float = 0.0
    status: str = "keep"  # keep | discard | crash
    parent_id: str | None = None
    origin: str = "seed"
    description: str = ""
    llm_rationale: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class Ledger:
    """Append-only JSONL ledger with frontier queries."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: Record) -> None:
        with open(self.path, "a") as fh:
            fh.write(rec.to_json() + "\n")

    def load(self) -> list[Record]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(Record(**json.loads(line)))
        return out

    def seen_ids(self) -> set[str]:
        return {r.id for r in self.load()}

    def kept(self) -> list[Record]:
        return [r for r in self.load() if r.status == "keep"]

    def best(self) -> Record | None:
        kept = self.kept()
        return max(kept, key=lambda r: r.metric) if kept else None

    def frontier(self, k: int = 8) -> list[Record]:
        """Top-K distinct kept candidates by metric (the breeding pool)."""
        seen: set[str] = set()
        uniq: list[Record] = []
        for r in sorted(self.kept(), key=lambda r: -r.metric):
            if r.id not in seen:
                seen.add(r.id)
                uniq.append(r)
        return uniq[:k]

    def frontier_candidates(self, k: int = 8) -> list[Candidate]:
        return [Candidate.from_dict(r.candidate) for r in self.frontier(k)]
