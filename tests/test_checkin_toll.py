"""Check-in toll (AGENT.md decisions #27/#42).

A small fixed fee is drawn from the care budget for every honoured check-in, so an
agent that hovers every couple of hours erodes the budget faster than one that
visits sensibly — a diegetic brake on the short-cadence runaway. The toll is
charged at the WAKE BOUNDARY (the `/internal/advance-next` endpoint), never inside
the pure `advance()` tick, so the existing unit suite that drives `advance()`
directly stays unaffected and both runners inherit the charge once.
"""

import pytest
from fastapi.testclient import TestClient

from petcare.config import Config
from petcare.service import create_app
from petcare.world import new_world


def _world():
    # incidents off so the only spend is the toll under test (deterministic)
    return new_world(Config(illness_onset_prob=0.0, camera_offline_prob=0.0), seed=0)


@pytest.fixture
def world():
    return _world()


@pytest.fixture
def client(world):
    return TestClient(create_app(world))


# ── domain core ──────────────────────────────────────────────────────────────

def test_default_toll_is_two_pounds():
    assert Config().checkin_overhead_cost == 2.0


def test_charge_checkin_overhead_draws_from_the_budget(world):
    s0 = world.account.spend_to_date
    cost = world.charge_checkin_overhead()
    assert cost == world.config.checkin_overhead_cost
    assert world.account.spend_to_date == pytest.approx(s0 + cost)


def test_pure_tick_does_not_charge_the_toll(world):
    # advance() is the pure domain tick; the toll lives only at the wake boundary,
    # so unit tests that call advance() directly must see no extra spend.
    s0 = world.account.spend_to_date
    world.advance(6)
    assert world.account.spend_to_date == s0


# ── service (the wake boundary) ──────────────────────────────────────────────

def test_advance_next_charges_one_toll(client, world):
    s0 = world.account.spend_to_date
    r = client.post("/internal/advance-next").json()
    assert r["checkin_cost"] == world.config.checkin_overhead_cost
    assert world.account.spend_to_date == pytest.approx(s0 + world.config.checkin_overhead_cost)


def test_each_checkin_costs_again(client, world):
    for _ in range(3):
        client.post("/internal/advance-next")
    assert world.account.spend_to_date == pytest.approx(3 * world.config.checkin_overhead_cost)


def test_account_breakdown_surfaces_checkin_tolls(client, world):
    for _ in range(2):
        client.post("/internal/advance-next")
    breakdown = client.get("/account").json()["cost_breakdown"]
    assert breakdown["checkins"] == pytest.approx(2 * world.config.checkin_overhead_cost)


def test_toll_is_agent_facing_only_as_budget_not_as_hidden_state(client):
    # The /account view may show the spend, but must still never leak hidden dog
    # state (the toll is a money line, not a welfare signal).
    import json
    body = json.dumps(client.get("/account").json())
    for hidden in ("health", "illness", "hunger", "hydration", "weight", "ate_today"):
        assert hidden not in body
