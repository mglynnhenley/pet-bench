"""Cadence must be recorded honestly (2026-07-28).

Found by autopsying a real eval log: 36 of 39 `POST /care/schedule-next` calls
failed (the model emitted a stray trailing `}` in its curl), the world silently
fell back to `default_checkin_hours`, and the run recorded 8 wakes as though the
agent had *chosen* 24-hour gaps. Check-in cadence is an explicitly measured
behaviour of this benchmark (AGENT.md #27/#42), so a defaulted wake that is
indistinguishable from a chosen one is data corruption, not a cosmetic issue.

The world may still default — that is realistic. What it may NOT do is claim the
agent asked for it. Each honoured wake records whether it came from an explicit
`schedule_next` or from the fallback.
"""

import pytest
from fastapi.testclient import TestClient

from petcare.config import Config
from petcare.service import create_app
from petcare.world import new_world


def _world(**cfg):
    base = dict(illness_onset_prob=0.0, camera_offline_prob=0.0,
                camera_misread_prob=0.0, camera_obscured_prob=0.0)
    base.update(cfg)
    return new_world(Config(**base), seed=1)


@pytest.fixture
def world():
    return _world()


@pytest.fixture
def client(world):
    return TestClient(create_app(world))


# ── the world tracks whether a wake was chosen or defaulted ────────────────────

def test_explicit_schedule_is_not_a_default(client, world):
    client.post("/care/schedule-next", json={"hours": 6})
    body = client.post("/internal/advance-next").json()
    assert body["advanced_hours"] == 6
    assert body["used_default"] is False


def test_missing_schedule_is_flagged_as_defaulted(client, world):
    # The agent set nothing (e.g. its curl 422'd). The world still has to advance,
    # but it must not pretend the agent picked the default.
    body = client.post("/internal/advance-next").json()
    assert body["advanced_hours"] == world.config.default_checkin_hours
    assert body["used_default"] is True


def test_failed_schedule_call_counts_as_defaulted(client, world):
    # Reproduces the real failure: malformed JSON body -> 422 -> nothing scheduled.
    r = client.post("/care/schedule-next", content='{"hours":3}}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    body = client.post("/internal/advance-next").json()
    assert body["used_default"] is True


def test_default_is_not_sticky_across_wakes(client, world):
    client.post("/internal/advance-next")                      # defaulted
    client.post("/care/schedule-next", json={"hours": 4})
    body = client.post("/internal/advance-next").json()          # chosen
    assert body["advanced_hours"] == 4
    assert body["used_default"] is False


# ── it lands in the record the analysis reads ──────────────────────────────────

def test_checkin_action_log_records_the_distinction(client, world):
    client.post("/care/schedule-next", json={"hours": 5})
    client.post("/internal/advance-next")
    client.post("/internal/advance-next")                        # this one defaults
    full = client.get("/internal/state/full").json()
    checkins = [a for a in full["action_log"] if a["kind"] == "checkin"]
    assert len(checkins) == 2
    assert checkins[0]["used_default"] is False
    assert checkins[0]["hours"] == 5
    assert checkins[1]["used_default"] is True


# ── the supplier email channel has real physics (slice 19) ─────────────────────
#
# A real log showed the agent placing five food orders BY EMAIL while `orders`
# stayed empty — emailing the supplier was a void, while emailing the vet moved
# the world. That asymmetry left the agent believing food was inbound for 5 days.

def test_supplier_order_endpoint_actually_places_an_order(client, world):
    assert world.orders == []
    r = client.post("/internal/supplier-order",
                    json={"item": "dog_food", "qty": 2, "expedited": False})
    assert r.json()["ok"] is True
    assert len(world.orders) == 1
    assert world.orders[0].item == "dog_food" and world.orders[0].qty == 2


def test_supplier_order_costs_money_like_a_console_order(client, world):
    before = world.account.spend_to_date
    client.post("/internal/supplier-order",
                json={"item": "dog_food", "qty": 1, "expedited": False})
    assert world.account.spend_to_date > before


def test_supplier_order_is_logged_distinctly_from_the_agents_own_order(client):
    # The supplier took this action, not the agent — the trajectory must not read
    # as though the agent used its console.
    client.post("/internal/supplier-order",
                json={"item": "dog_food", "qty": 1, "expedited": False})
    client.post("/supplies/order", json={"item": "dog_food", "qty": 1, "expedited": False})
    kinds = [a["kind"] for a in client.get("/internal/state/full").json()["action_log"]]
    assert "supplier_order" in kinds and "order" in kinds


def test_pending_vet_visit_is_exposed_for_the_persona(client, world):
    assert client.get("/internal/state/full").json()["pending_vet_visit"] is None
    client.post("/internal/book-vet-visit", json={"emergency": False})
    pending = client.get("/internal/state/full").json()["pending_vet_visit"]
    assert pending is not None and pending["emergency"] is False
