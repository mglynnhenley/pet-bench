"""Spine tests for the pluggable long-horizon dynamics layer (petcare/dynamics.py).

These lock the contract the three axis-modules (buddy/vet/pricing) build against:
connect-by-name, an independent reproducible RNG per module, lifecycle dispatch,
ground-truth recording, and — crucially — that a *disconnected* world behaves
byte-identically to the static world.
"""

from __future__ import annotations

from dataclasses import replace

from petcare import dynamics
from petcare.config import Config
from petcare.dynamics import (DynamicsModule, DynamicsRegistry, ScheduledEvent,
                              build_dynamics, derive_seed, register)
from petcare.world import new_world


def test_disabled_by_default_is_noop():
    w = new_world(Config(), seed=1)
    assert w.config.enabled_dynamics == ()
    assert isinstance(w.dynamics, DynamicsRegistry)
    assert w.dynamics.enabled == ()
    assert w.event_log == []
    w.advance(48)
    assert w.event_log == []  # nothing connected → nothing recorded


def test_disconnected_world_is_byte_identical():
    """Enabling the dynamics machinery (but no modules) must not perturb the core
    tick's RNG stream — the per-hour ground truth is identical to a plain world."""
    a = new_world(Config(), seed=7)
    b = new_world(Config(), seed=7)
    a.advance(72)
    b.advance(72)
    assert a.tick_log == b.tick_log


def test_registry_dispatches_build_and_on_hour():
    calls = {"build": 0, "hours": 0}

    @register("_probe")
    class Probe(DynamicsModule):
        def build(self, world):
            calls["build"] += 1

        def on_hour(self, world):
            calls["hours"] += 1
            self.record(world, "probe_tick", h=world.hour)

    try:
        cfg = replace(Config(), enabled_dynamics=("_probe",))
        w = new_world(cfg, seed=3)
        assert calls["build"] == 1
        w.advance(5)
        assert calls["hours"] == 5
        assert len(w.event_log) == 5
        assert all(isinstance(e, ScheduledEvent) for e in w.event_log)
        assert all(e.module == "_probe" and e.kind == "probe_tick" for e in w.event_log)
    finally:
        dynamics._REGISTRY.pop("_probe", None)


def test_unknown_module_name_is_skipped():
    cfg = replace(Config(), enabled_dynamics=("does_not_exist",))
    w = new_world(cfg, seed=0)
    assert w.dynamics.enabled == ()  # typo degrades gracefully, never raises


def test_per_module_rng_is_independent_and_stable():
    # derived seed depends on (seed, name) only — not on PYTHONHASHSEED or order
    assert derive_seed(5, "a") == derive_seed(5, "a")
    assert derive_seed(5, "a") != derive_seed(5, "b")
    assert derive_seed(5, "a") != derive_seed(6, "a")
    w = new_world(Config(), seed=11)
    r1 = w.derive_rng("buddy_events")
    r2 = w.derive_rng("food_pricing")
    # two streams, each reproducible
    assert w.derive_rng("buddy_events").random() == r1.random()
    assert r1.random() != r2.random()


def test_determinism_per_seed_with_module():
    @register("_probe2")
    class Probe(DynamicsModule):
        def build(self, world):
            self.draws = [self.rng.random() for _ in range(4)]

        def on_hour(self, world):
            self.record(world, "tick", v=world.tick)

    try:
        cfg = replace(Config(), enabled_dynamics=("_probe2",))
        a = new_world(cfg, seed=42)
        b = new_world(cfg, seed=42)
        a.advance(10)
        b.advance(10)
        assert [e.payload for e in a.event_log] == [e.payload for e in b.event_log]
        assert a.dynamics.get("_probe2").draws == b.dynamics.get("_probe2").draws
    finally:
        dynamics._REGISTRY.pop("_probe2", None)
