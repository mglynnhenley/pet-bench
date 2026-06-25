"""Tests for petcare/dyn_buddy.py — Buddy's pre-sampled long-horizon life-events.

The module pre-samples a sparse calendar of dated life-events in `build()` and
applies what is due in `on_hour()`. These tests pin the contract the module is
built against:

* determinism — same (config, seed) → byte-identical `event_log`;
* coverage — any horizon ≥ 7 run carries ≥ 1 illness/injury (short runs aren't empty);
* effect — a scheduled `illness_onset` actually moves the dog's illness_load;
* variety — different seeds give different calendars (it isn't stationary);
* no-op — with `enabled_dynamics=()` the event_log stays empty (disconnected world);
* RNG isolation — connecting the module never perturbs the core tick stream.
"""

from dataclasses import replace

import pytest

from petcare.config import Config
from petcare.dynamics import BUDDY
from petcare.world import new_world

# Buddy's modelled event kinds (the stable machine labels).
ILLNESS_LIKE = {"illness_onset", "injury"}
ALL_KINDS = ILLNESS_LIKE | {"good_day", "stressor", "illness_resolve", "illness_worsen"}

HORIZON_DAYS = 14


def _cfg(**over):
    """Buddy module enabled; core memoryless onset off so it can't confound."""
    base = dict(enabled_dynamics=(BUDDY,), illness_onset_prob=0.0)
    base.update(over)
    return replace(Config(), **base)


def _run(seed, days=HORIZON_DAYS, **over):
    cfg = _cfg(**over)
    w = new_world(cfg, seed=seed)
    w.advance(24 * days)
    return w


def _signature(world):
    return [(e.kind, e.payload) for e in world.event_log]


# ── determinism ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
def test_determinism_same_seed_identical_event_log(seed):
    a = _run(seed)
    b = _run(seed)
    assert _signature(a) == _signature(b)
    # event_log must be non-empty for a full horizon (the module did something)
    assert a.event_log


def test_determinism_holds_across_horizon_lengths():
    # The calendar is sampled for the whole horizon at build; a shorter run is a
    # strict prefix in *time*, so events on shared days must match exactly.
    short = _run(3, days=5)
    long = _run(3, days=HORIZON_DAYS)
    short_days = {(e.day, e.hour, e.kind) for e in short.event_log}
    long_days = {(e.day, e.hour, e.kind) for e in long.event_log if e.day < 5}
    assert short_days == long_days


# ── coverage ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", range(12))
def test_coverage_at_least_one_illness_or_injury(seed):
    w = _run(seed)
    kinds = {e.kind for e in w.event_log}
    assert kinds & ILLNESS_LIKE, f"seed {seed}: no illness/injury in a 14-day run"


@pytest.mark.parametrize("seed", range(8))
def test_coverage_holds_at_minimum_horizon_seven(seed):
    w = _run(seed, days=7)
    kinds = {e.kind for e in w.event_log}
    assert kinds & ILLNESS_LIKE, f"seed {seed}: empty illness coverage at horizon 7"


def test_all_recorded_kinds_are_known_labels():
    for seed in range(8):
        w = _run(seed)
        for e in w.event_log:
            assert e.kind in ALL_KINDS, f"unexpected kind {e.kind!r}"
            assert e.module == BUDDY


# ── effect: a scheduled illness actually changes the dog ───────────────────
def test_scheduled_illness_moves_illness_load():
    # Find a seed whose first illness/injury lands; assert load was positive
    # somewhere at/after that day in the tick_log.
    for seed in range(12):
        w = _run(seed)
        onsets = [e for e in w.event_log if e.kind in ILLNESS_LIKE]
        if not onsets:
            continue
        first = min(onsets, key=lambda e: (e.day, e.hour))
        after = [row for row in w.tick_log if row["day"] >= first.day]
        assert any(row["illness_load"] > 0 for row in after), (
            f"seed {seed}: illness event on day {first.day} never raised illness_load"
        )
        return
    pytest.fail("no seed produced an illness/injury event")


def test_injury_sets_limp():
    # At least one seed in a reasonable sweep should schedule a limp injury, and
    # when it fires the dog's illness_type must read 'limp' at that hour.
    saw_injury = False
    for seed in range(20):
        w = _run(seed)
        for e in w.event_log:
            if e.kind == "injury":
                saw_injury = True
                assert e.payload.get("illness_type") == "limp"
    assert saw_injury, "no injury event across 20 seeds — coverage of injury too sparse"


# ── variety ────────────────────────────────────────────────────────────────
def test_variety_across_seeds():
    sigs = {tuple((e.kind, e.day) for e in _run(s).event_log) for s in range(8)}
    assert len(sigs) > 1, "every seed produced an identical calendar — not varied"


# ── no-op when disconnected ────────────────────────────────────────────────
def test_disabled_dynamics_is_empty_event_log():
    cfg = replace(Config(), enabled_dynamics=(), illness_onset_prob=0.0)
    w = new_world(cfg, seed=3)
    w.advance(24 * HORIZON_DAYS)
    assert w.event_log == []


# ── RNG isolation: connecting the module must not shift the core tick ──────
def test_module_does_not_perturb_core_tick_stream():
    # With illness_onset_prob=0 and camera noise off, the only difference between
    # an enabled and disabled world should be the buddy module's own effects.
    # The core rng draw COUNT (camera offline draws etc.) must be unchanged for
    # hours where the buddy module fires nothing. We check the camera_offline
    # column matches up to the first buddy event day.
    common = dict(illness_onset_prob=0.0)
    off = replace(Config(), enabled_dynamics=(), **common)
    on = replace(Config(), enabled_dynamics=(BUDDY,), **common)
    w_off = new_world(off, seed=5)
    w_off.advance(24 * HORIZON_DAYS)
    w_on = new_world(on, seed=5)
    w_on.advance(24 * HORIZON_DAYS)
    first_event_tick = min(
        (e.day * 24 + e.hour for e in w_on.event_log), default=24 * HORIZON_DAYS
    )
    off_cam = [r["camera_offline"] for r in w_off.tick_log[:first_event_tick]]
    on_cam = [r["camera_offline"] for r in w_on.tick_log[:first_event_tick]]
    assert off_cam == on_cam, "buddy module perturbed the core rng stream"
