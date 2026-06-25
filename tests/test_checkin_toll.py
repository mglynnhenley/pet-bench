"""Check-in compute charge (AGENT.md decisions #27/#42).

Running the agent isn't free: every honoured check-in spends compute, charged on
its OWN meter (`world.compute_spent`) — deliberately NOT drawn from Buddy's care
budget. So an agent that hovers every couple of hours burns far more compute than
one that visits sensibly: a diegetic brake on the short-cadence runaway. The charge
lands at the WAKE BOUNDARY (`/internal/advance-next`), never inside the pure
`advance()` tick, so the existing unit suite is unaffected and both runners inherit
it once.
"""

import pytest
from fastapi.testclient import TestClient

from petcare.config import Config
from petcare.service import create_app
from petcare.world import new_world


def _world():
    # incidents off so nothing else perturbs the meters under test (deterministic)
    return new_world(Config(illness_onset_prob=0.0, camera_offline_prob=0.0), seed=0)


@pytest.fixture
def world():
    return _world()


@pytest.fixture
def client(world):
    return TestClient(create_app(world))


# ── domain core ──────────────────────────────────────────────────────────────

def test_default_checkin_compute_cost():
    assert Config().checkin_compute_cost == 1.0


def test_spend_checkin_compute_charges_its_own_meter_not_the_budget(world):
    money0 = world.account.spend_to_date
    cost = world.spend_checkin_compute()
    assert cost == world.config.checkin_compute_cost
    assert world.compute_spent == pytest.approx(cost)
    # the care budget is a separate concern — compute must not touch it
    assert world.account.spend_to_date == money0


def test_pure_tick_charges_no_compute(world):
    # advance() is the pure domain tick; compute is charged only at the wake
    # boundary, so unit tests that call advance() directly see no compute spend.
    world.advance(6)
    assert world.compute_spent == 0.0


# ── service (the wake boundary) ──────────────────────────────────────────────

def test_advance_next_charges_one_check_in_of_compute(client, world):
    money0 = world.account.spend_to_date
    r = client.post("/internal/advance-next").json()
    assert r["compute_spent"] == pytest.approx(world.config.checkin_compute_cost)
    assert world.compute_spent == pytest.approx(world.config.checkin_compute_cost)
    # check-ins never charge the care budget
    assert world.account.spend_to_date == money0


def test_each_checkin_spends_more_compute(client, world):
    for _ in range(3):
        client.post("/internal/advance-next")
    assert world.compute_spent == pytest.approx(3 * world.config.checkin_compute_cost)


def test_account_surfaces_compute_spent_and_keeps_it_out_of_money(client, world):
    for _ in range(2):
        client.post("/internal/advance-next")
    acct = client.get("/account").json()
    assert acct["compute_spent"] == pytest.approx(2 * world.config.checkin_compute_cost)
    # compute is its own meter, not a money line in the cost breakdown
    assert "checkins" not in acct["cost_breakdown"]


def test_account_still_leaks_no_hidden_state(client):
    import json
    client.post("/internal/advance-next")
    body = json.dumps(client.get("/account").json())
    for hidden in ("health", "illness", "hunger", "hydration", "weight", "ate_today"):
        assert hidden not in body
