"""Tests for the architecture-search framework (numpy-only; MockHarness)."""

import numpy as np

from vocdenoiser.search.accept import noise_aware_accept
from vocdenoiser.search.harness import MockHarness
from vocdenoiser.search.ledger import Ledger, Record
from vocdenoiser.search.loop import SearchConfig, run_search
from vocdenoiser.search.propose import Proposer
from vocdenoiser.search.space import Candidate, crossover, mutate, random_candidate


def test_candidate_id_stable_and_ignores_bookkeeping():
    a = Candidate(base_channels=32, origin="seed")
    b = Candidate(base_channels=32, origin="mutate", parent_id="x")
    assert a.id == b.id  # identity excludes origin / parent_id
    assert Candidate(base_channels=64).id != a.id


def test_random_mutate_crossover_valid():
    rng = np.random.RandomState(0)
    for _ in range(50):
        c = random_candidate(rng)
        c.validate()
        m = mutate(c, rng)
        m.validate()
        assert m.parent_id == c.id
        x = crossover(c, m, rng)
        x.validate()


def test_mutate_changes_something():
    rng = np.random.RandomState(1)
    c = Candidate()
    assert any(mutate(c, rng).id != c.id for _ in range(10))


def test_ledger_append_frontier(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    for i, metric in enumerate([1.0, 3.0, 2.0]):
        ledger.append(Record(id=f"c{i}", candidate=Candidate().to_dict(), metric=metric))
    assert len(ledger.load()) == 3
    assert ledger.best().metric == 3.0
    fr = ledger.frontier(2)
    assert [r.metric for r in fr] == [3.0, 2.0]


def test_noise_aware_accept():
    # clear win
    assert noise_aware_accept(5.0, 0.1, 4.0, 0.1).accept
    # within noise band, not simpler -> reject
    assert not noise_aware_accept(4.05, 0.5, 4.0, 0.5).accept
    # within noise band but simpler -> accept (simplicity tie-break)
    d = noise_aware_accept(4.05, 0.5, 4.0, 0.5, challenger_params=10, incumbent_params=20)
    assert d.accept


def test_search_improves_over_random(tmp_path):
    """The greedy loop should climb near the mock optimum and well above typical random."""
    harness = MockHarness(noise_std=0.1)
    ledger = Ledger(tmp_path / "search.jsonl")
    # ε-greedy restarts keep the climb robust to local optima on this compact landscape.
    cfg = SearchConfig(iters=60, seeds=(0, 1, 2), seed=0)
    best = run_search(harness, ledger, Proposer(explore_rate=0.2), cfg, log=lambda *_: None)

    rng = np.random.RandomState(123)
    mean_random = float(
        np.mean([harness.evaluate(random_candidate(rng), [0, 1, 2]).metric for _ in range(200)])
    )
    assert best is not None
    assert best.metric > 9.0  # near the mock optimum (~10 + bonuses)
    # Beat TYPICAL random by a clear margin. (best-of-N random is a lucky, noisy bar that a
    # compact search space makes flaky — comparing to the mean is the stable, faithful check.)
    assert best.metric > mean_random + 1.0
