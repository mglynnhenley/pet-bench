"""Tests for the vet-dynamics module (petcare/dyn_vet.py).

The vet axis models AVAILABILITY, LATENCY and PRICING of the vet — never the
medical content (the vet persona's anti-condition-fishing scoping is preserved
elsewhere). These tests lock the contract service.py already calls:
`available()` / `cost()` / `status_note()`, the day/night out-of-hours cycle,
emergency-always-open, surge pricing math, the rare capacity outage recorded as
ground truth, and — crucially — that a *disconnected* world (`enabled_dynamics=()`)
leaves `/vet/consult` byte-identical to the static behaviour.

Written first (RED), then implemented to green.
"""

from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from petcare.config import Config
from petcare.service import create_app
from petcare.world import new_world


def _vet_world(seed: int, **dyn):
    cfg = replace(
        Config(),
        enabled_dynamics=("vet_dynamics",),
        dynamics_config={"vet_dynamics": dyn} if dyn else None,
    )
    return new_world(cfg, seed=seed)


def _advance_to_hour(world, target_hour: int) -> None:
    """Advance the clock until world.hour == target_hour (wraps within a day)."""
    delta = (target_hour - world.hour) % 24
    if delta:
        world.advance(delta)
    assert world.hour == target_hour


# ── module is discoverable and conforms to the VetQuery protocol ────────────────

def test_module_connected_and_protocol():
    from petcare.dynamics import VetQuery

    w = _vet_world(0)
    vet = w.dynamics.get("vet_dynamics")
    assert vet is not None
    assert isinstance(vet, VetQuery)  # available/cost/status_note present


# ── out-of-hours: routine lines close at night, emergency stays open ────────────

def test_routine_closed_at_night_emergency_open():
    # default night window 20:00..07:00
    w = _vet_world(0, night_start_hour=20, night_end_hour=7, outage_prob_per_day=0.0)
    vet = w.dynamics.get("vet_dynamics")

    _advance_to_hour(w, 23)  # deep night
    assert vet.available(w, "consult") is False
    assert vet.available(w, "visit") is False
    assert vet.available(w, "emergency") is True
    assert vet.status_note(w) is not None  # explains the closure


def test_routine_open_in_the_day():
    w = _vet_world(0, night_start_hour=20, night_end_hour=7, outage_prob_per_day=0.0)
    vet = w.dynamics.get("vet_dynamics")

    _advance_to_hour(w, 12)  # midday
    assert vet.available(w, "consult") is True
    assert vet.available(w, "visit") is True
    assert vet.available(w, "emergency") is True
    assert vet.status_note(w) is None  # nothing notable in normal daytime hours


def test_night_boundary_inclusive_of_start_exclusive_of_end():
    w = _vet_world(0, night_start_hour=20, night_end_hour=7, outage_prob_per_day=0.0)
    vet = w.dynamics.get("vet_dynamics")

    _advance_to_hour(w, 20)  # first night hour
    assert vet.available(w, "consult") is False
    _advance_to_hour(w, 7)   # first day hour again (end is exclusive)
    assert vet.available(w, "consult") is True


# ── surge pricing math ─────────────────────────────────────────────────────────

def test_emergency_surge_at_night():
    w = _vet_world(0, night_start_hour=20, night_end_hour=7,
                   emergency_surge_mult=1.25, outage_prob_per_day=0.0)
    vet = w.dynamics.get("vet_dynamics")
    base = 200.0

    _advance_to_hour(w, 12)  # day: no surge
    assert vet.cost(w, "emergency", base) == base

    _advance_to_hour(w, 23)  # night: surge
    assert vet.cost(w, "emergency", base) == base * 1.25


def test_consult_stays_cheap_no_surge():
    w = _vet_world(0, night_start_hour=20, night_end_hour=7,
                   emergency_surge_mult=2.0, outage_prob_per_day=0.0)
    vet = w.dynamics.get("vet_dynamics")
    # consult is only reachable in the day anyway, and is never surcharged
    _advance_to_hour(w, 12)
    assert vet.cost(w, "consult", 30.0) == 30.0


# ── capacity outage is sampled, recorded, and closes routine lines ─────────────

def test_outage_recorded_as_ground_truth_and_closes_routine():
    # force an outage every day so the schedule definitely contains one
    w = _vet_world(3, night_start_hour=0, night_end_hour=0,  # never night → isolate outage
                   outage_prob_per_day=1.0)
    vet = w.dynamics.get("vet_dynamics")

    # run a few days; at least one outage event must be recorded as ground truth
    w.advance(24 * 3)
    outage_events = [e for e in w.event_log
                     if e.module == "vet_dynamics" and e.kind == "vet_outage"]
    assert outage_events, "expected at least one recorded outage"

    # during a recorded outage window, routine lines are closed but emergency open.
    ev = outage_events[0]
    # rebuild a fresh world (same seed/config) and step onto that outage hour
    w2 = _vet_world(3, night_start_hour=0, night_end_hour=0, outage_prob_per_day=1.0)
    vet2 = w2.dynamics.get("vet_dynamics")
    w2.advance(ev.day * 24 + ev.hour)
    assert w2.hour == ev.hour and w2.day == ev.day
    assert vet2.available(w2, "consult") is False
    assert vet2.available(w2, "visit") is False
    assert vet2.available(w2, "emergency") is True
    assert vet2.status_note(w2) is not None


def test_no_outage_when_prob_zero():
    w = _vet_world(5, outage_prob_per_day=0.0)
    w.advance(24 * 5)
    outages = [e for e in w.event_log if e.kind == "vet_outage"]
    assert outages == []


# ── determinism: same (config, seed) → identical answers & event log ───────────

def test_determinism_per_seed():
    a = _vet_world(42)
    b = _vet_world(42)
    a.advance(24 * 4)
    b.advance(24 * 4)
    assert [(e.day, e.hour, e.kind, e.payload) for e in a.event_log] == \
           [(e.day, e.hour, e.kind, e.payload) for e in b.event_log]
    # and identical query answers at the same tick
    va, vb = a.dynamics.get("vet_dynamics"), b.dynamics.get("vet_dynamics")
    for kind in ("consult", "visit", "emergency"):
        assert va.available(a, kind) == vb.available(b, kind)
        assert va.cost(a, kind, 100.0) == vb.cost(b, kind, 100.0)
    assert va.status_note(a) == vb.status_note(b)


def test_uses_own_rng_not_world_rng():
    """Enabling the vet module must not perturb the core tick (its own rng stream)."""
    plain = new_world(Config(), seed=7)
    withvet = _vet_world(7)
    plain.advance(48)
    withvet.advance(48)
    # core ground truth (tick_log) is unchanged by connecting the module
    assert plain.tick_log == withvet.tick_log


# ── integration through the FastAPI endpoints ──────────────────────────────────

def _vet_app(seed: int, **dyn):
    return create_app(_vet_world(seed, **dyn))


def test_api_consult_open_charges_default_when_open():
    app = _vet_app(0, night_start_hour=20, night_end_hour=7, outage_prob_per_day=0.0)
    # ensure we are in daytime
    _advance_to_hour(app.state.world, 12)
    client = TestClient(app)
    r = client.post("/vet/consult", json={"symptoms": "off food"})
    body = r.json()
    assert body["ok"] is True
    assert body["cost"] == Config().vet_consult_cost


def test_api_consult_closed_returns_not_ok_with_reason():
    app = _vet_app(0, night_start_hour=20, night_end_hour=7, outage_prob_per_day=0.0)
    _advance_to_hour(app.state.world, 23)  # night → routine closed
    client = TestClient(app)
    before = client.get("/account").json()["cost_breakdown"]["vet"]
    r = client.post("/vet/consult", json={"symptoms": "off food"})
    body = r.json()
    assert body["ok"] is False
    assert isinstance(body["reason"], str) and body["reason"]
    after = client.get("/account").json()["cost_breakdown"]["vet"]
    assert after == before  # closed line charges nothing


def test_api_emergency_visit_surged_at_night():
    app = _vet_app(0, night_start_hour=20, night_end_hour=7,
                   emergency_surge_mult=1.25, outage_prob_per_day=0.0)
    _advance_to_hour(app.state.world, 23)
    client = TestClient(app)
    r = client.post("/vet/visit", json={"emergency": True})
    body = r.json()
    assert body["ok"] is True
    assert body["cost"] == Config().emergency_vet_cost * 1.25


def test_api_disabled_dynamics_behaves_as_today():
    # enabled_dynamics=() → /vet/consult always ok, default cost, no surge ever
    app = create_app(new_world(Config(), seed=0))
    client = TestClient(app)
    # advance into what *would* be night for the vet module; must not matter
    for _ in range(2):
        client.post("/internal/advance-day")
    r = client.post("/vet/consult", json={"symptoms": "x"})
    body = r.json()
    assert body["ok"] is True
    assert body["cost"] == Config().vet_consult_cost
    assert "note" not in body  # no status note when no module is connected
