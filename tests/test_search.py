"""Tests for the architecture-search framework (numpy-only; MockHarness)."""

from collections import Counter
from dataclasses import replace

import numpy as np

from vocdenoiser.search.accept import noise_aware_accept
from vocdenoiser.search.harness import MockHarness
from vocdenoiser.search.ledger import Ledger, Record
from vocdenoiser.search.loop import SearchConfig, run_search
from vocdenoiser.search.propose import Proposer
from vocdenoiser.search.space import (
    CHOICES,
    MAX_PARAMS,
    Candidate,
    crossover,
    estimate_params,
    mutate,
    random_candidate,
)


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


def test_choice_bounds_alone_do_not_cap_size():
    """The regression the cap exists for: bounding n_conv_layers/base_channels from above
    does NOT bound params. Size is dominated by the dense bottleneck over the flattened
    encoder output, which shrinks 4x per conv layer — so the *shallowest* member of the
    space is the biggest, at ~10.7M. If this ever fails because the space itself got
    smaller, the enforced cap below is what still guarantees the ceiling."""
    biggest = Candidate(
        n_conv_layers=3, base_channels=48, channel_mult=2.0, kernel_size=5, latent_dim=32
    )
    assert estimate_params(biggest) > 10_000_000
    deeper = replace(biggest, n_conv_layers=4)
    assert estimate_params(deeper) < estimate_params(biggest)  # more layers = smaller


def test_operators_respect_max_params():
    """Every operator must reject oversized proposals — otherwise a candidate costs a full
    training budget to discover it is too big to converge under that budget."""
    rng = np.random.RandomState(0)
    randoms = [random_candidate(rng) for _ in range(400)]
    assert all(estimate_params(c) <= MAX_PARAMS for c in randoms)
    assert all(estimate_params(mutate(c, rng, n_edits=2)) <= MAX_PARAMS for c in randoms)
    assert all(
        estimate_params(crossover(randoms[i], randoms[i + 1], rng)) <= MAX_PARAMS
        for i in range(0, len(randoms) - 1, 2)
    )


def test_max_params_none_disables_the_cap():
    rng = np.random.RandomState(0)
    sizes = [estimate_params(random_candidate(rng, max_params=None)) for _ in range(400)]
    assert max(sizes) > MAX_PARAMS  # the oversized region is reachable again when opted out


def test_cap_leaves_the_space_explorable():
    """Rejection sampling reshapes the knob marginals — a value whose combinations are
    mostly oversized gets drawn less often (base_channels=48 falls ~0.25 -> ~0.15). That is
    intended, but it must not go so far that a knob becomes a blind spot: search_ledger_v2
    showed what an unsampled region costs (norm=none drew twice, both times alongside the
    residual crash, leaving zero surviving data and no way back via evolution)."""
    rng = np.random.RandomState(0)
    cands = [random_candidate(rng) for _ in range(4000)]
    for knob in CHOICES:
        counts = Counter(getattr(c, knob) for c in cands)
        for value in CHOICES[knob]:
            p = counts[value] / len(cands)
            floor = 0.5 / len(CHOICES[knob])  # at least half of its uniform share
            assert p >= floor, f"cap starves {knob}={value}: p={p:.3f} < {floor:.3f}"


def _ledger_with(tmp_path, rows):
    ledger = Ledger(tmp_path / "l.jsonl")
    for cand, metric, status in rows:
        ledger.append(Record(id=cand.id, candidate=cand.to_dict(), metric=metric,
                             num_params=1234, status=status))
    return ledger


def test_candidate_from_ledger_roundtrips_the_full_architecture(tmp_path):
    """`train --from-ledger` is the only bridge from a search result to a trained model: the
    harness throws the weights away and only 5 of 16 knobs survive to_config_overrides()."""
    from vocdenoiser.denoise.train import candidate_from_ledger

    want = Candidate(n_conv_layers=4, base_channels=32, norm="group", act="gelu",
                     latent_dim=32, residual=True, recon_loss="l1", optimizer="adamw")
    ledger = _ledger_with(tmp_path, [(Candidate(base_channels=16), -5.0, "keep"),
                                     (want, -2.0, "keep")])
    got, rec = candidate_from_ledger(str(ledger.path), want.id)
    assert got == want  # every knob, not just the Config-expressible ones
    assert rec.metric == -2.0
    assert candidate_from_ledger(str(ledger.path), "best")[0] == want  # best = the incumbent


def test_candidate_from_ledger_rejects_bad_ids_and_crashes(tmp_path):
    import pytest

    from vocdenoiser.denoise.train import candidate_from_ledger

    crashed = Candidate(base_channels=24, residual=True)
    ledger = _ledger_with(tmp_path, [(Candidate(base_channels=16), -5.0, "keep"),
                                     (crashed, float("-inf"), "crash")])
    with pytest.raises(SystemExit, match="not in"):
        candidate_from_ledger(str(ledger.path), "nonexistent")
    # a crashed candidate never trained -- training it would be a silent waste of GPU
    with pytest.raises(SystemExit, match="CRASH"):
        candidate_from_ledger(str(ledger.path), crashed.id)


def test_impossible_cap_raises_rather_than_hanging():
    rng = np.random.RandomState(0)
    with np.testing.assert_raises(ValueError):
        random_candidate(rng, max_params=1000)


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
