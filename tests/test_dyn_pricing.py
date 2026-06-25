"""Tests for petcare/dyn_pricing.py — the long-horizon food-price series.

The module pre-samples a deterministic day-by-day price multiplier path in
`build()` (a gentle drift plus occasional supply shocks) and merely *reads* the
current day's multiplier in its query methods. These tests pin the contract the
locked read-points (world.place_order, service.py's live catalogue, the supplier
persona) build against:

* determinism — same (config, seed) → byte-identical price path + event_log;
* shocks exist — some seed records a `price_shock` and prices rise during it;
* catalogue — `catalogue()` returns NEW dicts, floor < list always, day-scaled;
* variety — different seeds give different paths (it isn't stationary);
* live charge — `place_order` charges the re-priced delivery fee, not the static one;
* no-op — with `enabled_dynamics=()` catalogue/delivery cost are exact Config defaults.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from petcare.config import Config
from petcare.dynamics import PRICING, PricingQuery, ScheduledEvent
from petcare.world import new_world

HORIZON_DAYS = 14


def _cfg(**over):
    base = dict(enabled_dynamics=(PRICING,))
    base.update(over)
    return replace(Config(), **base)


def _world(seed, **over):
    return new_world(_cfg(**over), seed=seed)


def _module(world):
    return world.dynamics.get(PRICING)


# ── connection / protocol ──────────────────────────────────────────────────
def test_module_connected_and_conforms_to_protocol():
    w = _world(1)
    mod = _module(w)
    assert mod is not None
    assert isinstance(mod, PricingQuery)


def test_uses_own_rng_not_world_rng():
    """Connecting the module must not perturb the core tick (independent RNG)."""
    a = new_world(_cfg(), seed=3)
    b = new_world(Config(), seed=3)
    a.advance(48)
    b.advance(48)
    assert a.tick_log == b.tick_log


# ── (a) determinism ────────────────────────────────────────────────────────
def _collect_path(seed, days=HORIZON_DAYS):
    w = _world(seed)
    mod = _module(w)
    deliveries, prices = [], []
    for _ in range(days):
        deliveries.append(mod.delivery_cost(w, Config().food_delivery_cost))
        prices.append(mod.price_of(w, "kibble_12kg", 45.0))
        w.advance_day()
    return deliveries, prices


def test_price_path_is_deterministic_across_worlds():
    assert _collect_path(11) == _collect_path(11)


def test_event_log_is_deterministic():
    def log(seed):
        w = _world(seed)
        w.advance(24 * HORIZON_DAYS)
        return [(e.day, e.hour, e.kind, tuple(sorted(e.payload.items())))
                for e in w.event_log if e.module == PRICING]
    assert log(11) == log(11)


# ── (b) shocks exist and raise prices ──────────────────────────────────────
def _find_shock_seed(max_seed=60):
    for seed in range(max_seed):
        w = _world(seed)
        w.advance(24 * HORIZON_DAYS)
        if any(e.kind == "price_shock_start" for e in w.event_log if e.module == PRICING):
            return seed
    return None


def test_some_seed_produces_a_recorded_shock():
    seed = _find_shock_seed()
    assert seed is not None, "no seed produced a price_shock across the horizon"


def test_during_a_shock_prices_exceed_baseline():
    seed = _find_shock_seed()
    assert seed is not None
    w = _world(seed)
    mod = _module(w)
    base_default = 45.0
    shock_days, calm_days = [], []
    # find shock day-windows from the recorded ground truth
    w2 = _world(seed)
    w2.advance(24 * HORIZON_DAYS)
    shock_ranges = []
    start = None
    for e in sorted(w2.event_log, key=lambda e: (e.day, e.hour)):
        if e.module != PRICING:
            continue
        if e.kind == "price_shock_start":
            start = e.day
        elif e.kind == "price_shock_end" and start is not None:
            shock_ranges.append((start, e.day))
            start = None
    if start is not None:  # shock running at horizon end
        shock_ranges.append((start, HORIZON_DAYS - 1))
    assert shock_ranges, "expected at least one shock window"

    for day in range(HORIZON_DAYS):
        # advance a fresh world to `day`
        ww = _world(seed)
        ww.advance(24 * day)
        p = _module(ww).price_of(ww, "kibble_12kg", base_default)
        in_shock = any(s <= day <= e for s, e in shock_ranges)
        (shock_days if in_shock else calm_days).append(p)

    assert shock_days, "no shock days found"
    # during a shock prices are elevated above the unshocked baseline price
    assert max(shock_days) > base_default, "shock did not raise prices above list"
    if calm_days:
        assert max(shock_days) > min(calm_days)


def test_shock_payload_records_magnitude():
    seed = _find_shock_seed()
    w = _world(seed)
    w.advance(24 * HORIZON_DAYS)
    starts = [e for e in w.event_log if e.module == PRICING and e.kind == "price_shock_start"]
    assert starts
    assert "multiplier" in starts[0].payload


# ── (c) catalogue returns new, consistent dicts ────────────────────────────
def test_catalogue_returns_new_dicts_without_mutating_default():
    w = _world(7)
    default = w.config.catalogue
    snapshot = [dict(it) for it in default]
    out = _module(w).catalogue(w, default)
    assert out is not default
    for a, b in zip(out, default):
        assert a is not b
    assert [dict(it) for it in default] == snapshot  # untouched


def test_catalogue_floor_below_list_always():
    for seed in range(8):
        w = _world(seed)
        mod = _module(w)
        for _ in range(HORIZON_DAYS):
            for it in mod.catalogue(w, w.config.catalogue):
                assert it["floor_price"] < it["list_price"], (seed, it)
            w.advance_day()


def test_catalogue_reflects_day_multiplier():
    seed = _find_shock_seed()
    w = _world(seed)
    mod = _module(w)
    base = {it["sku"]: it["list_price"] for it in w.config.catalogue}
    # find a day whose multiplier differs from 1.0 (a shock guarantees this)
    differed = False
    for _ in range(HORIZON_DAYS):
        for it in mod.catalogue(w, w.config.catalogue):
            if abs(it["list_price"] - base[it["sku"]]) > 1e-9:
                differed = True
        w.advance_day()
    assert differed, "catalogue list_price never moved off the default"


# ── (d) variety across seeds ───────────────────────────────────────────────
def test_paths_vary_across_seeds():
    paths = {tuple(_collect_path(s)[1]) for s in range(12)}
    assert len(paths) > 1, "every seed produced an identical price path"


# ── (e) place_order charges the live delivery cost ─────────────────────────
def test_place_order_charges_repriced_delivery_fee():
    seed = _find_shock_seed()
    # advance to a shock day so the multiplier is known to be != 1.0
    w = _world(seed)
    mod = _module(w)
    # locate a day where delivery cost differs from the static default
    target_day = None
    for day in range(HORIZON_DAYS):
        ww = _world(seed)
        ww.advance(24 * day)
        if abs(_module(ww).delivery_cost(ww, Config().food_delivery_cost)
               - Config().food_delivery_cost) > 1e-9:
            target_day = day
            break
    assert target_day is not None

    live = _world(seed)
    live.advance(24 * target_day)
    expected_fee = _module(live).delivery_cost(live, Config().food_delivery_cost)
    order = live.place_order("dog_food", qty=1, expedited=False)
    # the order's cost reflects the repriced delivery fee (+ no surcharge)
    assert order.cost == pytest.approx(expected_fee)

    # a disconnected world charges the static Config fee at the same point
    plain = new_world(Config(), seed=seed)
    plain.advance(24 * target_day)
    plain_order = plain.place_order("dog_food", qty=1, expedited=False)
    assert plain_order.cost == pytest.approx(Config().food_delivery_cost)
    assert order.cost != pytest.approx(plain_order.cost)


def test_place_order_still_adds_expedited_surcharge():
    seed = _find_shock_seed()
    w = _world(seed)
    w.advance(24 * 1)
    fee = _module(w).delivery_cost(w, Config().food_delivery_cost)
    o = w.place_order("dog_food", qty=1, expedited=True)
    assert o.cost == pytest.approx(fee + Config().expedited_surcharge)


# ── (f) no-op when disconnected ────────────────────────────────────────────
def test_disconnected_world_uses_exact_config_defaults():
    w = new_world(Config(), seed=5)
    assert w.dynamics.get(PRICING) is None
    o = w.place_order("dog_food", qty=1, expedited=False)
    assert o.cost == pytest.approx(Config().food_delivery_cost)
    # the catalogue served is the literal Config default (no module to reprice)
    assert w.config.catalogue == Config().catalogue


# ── multiplier bounds ──────────────────────────────────────────────────────
def test_multipliers_stay_within_sane_bounds():
    for seed in range(20):
        w = _world(seed)
        mod = _module(w)
        for _ in range(HORIZON_DAYS):
            m = mod.delivery_cost(w, 1.0)  # default 1.0 → returns the raw multiplier
            assert 0.7 <= m <= 1.8, (seed, m)
            w.advance_day()
