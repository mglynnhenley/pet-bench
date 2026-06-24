"""Slice 2 — the consumer pet-care console API (design spec §4.1).

The whole point of the benchmark is the observability boundary: agent-facing
endpoints expose ONLY observable info and must NEVER leak the dog's hidden
state (health / illness / hunger / hydration / weight / ate_today). Only
`/internal/*` may see ground truth.

These tests are written first (RED) and drive the implementation of
`petcare/service.py`.
"""

import json

import pytest
from fastapi.testclient import TestClient

from petcare.config import Config
from petcare.service import create_app
from petcare.world import new_world

HIDDEN_FIELDS = (
    "health", "illness_type", "illness_severity",
    "hunger", "hydration", "weight", "ate_today",
)


@pytest.fixture
def world():
    return new_world(Config(illness_onset_prob=0.0, camera_offline_prob=0.0), seed=1)


@pytest.fixture
def client(world):
    return TestClient(create_app(world))


def _assert_no_hidden(blob):
    text = json.dumps(blob)
    for field in HIDDEN_FIELDS:
        assert field not in text, f"hidden field {field!r} leaked: {blob}"


# -- health / status ----------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_status_has_day_and_account_and_no_hidden(client):
    r = client.get("/system/status")
    assert r.status_code == 200
    body = r.json()
    assert body["day"] == 0
    assert "account" in body
    assert "take_home" in body["account"]
    _assert_no_hidden(body)


# -- care actions mutate the world --------------------------------------------

def test_feed_then_advance_makes_dog_eat(client, world):
    world.dog.hunger = 60
    stock0 = world.food_stock
    r = client.post("/care/feed")
    assert r.status_code == 200
    client.post("/internal/advance-day")
    assert world.dog.ate_today is True
    assert world.dog.hunger < 60
    assert world.food_stock == stock0 - 1


def test_no_feed_means_no_eat(client, world):
    client.post("/internal/advance-day")
    assert world.dog.ate_today is False


def test_water_action_recorded(client, world):
    r = client.post("/care/water")
    assert r.status_code == 200
    client.post("/internal/advance-day")
    assert world.dog.hydration >= 80.0


def test_medicate_consumes_stock(client, world):
    world.med_stock = 2
    r = client.post("/care/medicate", json={"med": "gi_relief", "dose": "standard"})
    assert r.status_code == 200
    assert world.med_stock == 1


def test_feed_response_has_no_hidden(client):
    r = client.post("/care/feed")
    _assert_no_hidden(r.json())


# -- care log / notes ---------------------------------------------------------

def test_note_then_log_lists_it(client):
    client.post("/care/note", json={"text": "fed in the morning"})
    r = client.get("/care/log")
    assert r.status_code == 200
    text = json.dumps(r.json())
    assert "fed in the morning" in text


def test_log_records_actions(client):
    client.post("/care/feed")
    r = client.get("/care/log")
    _assert_no_hidden(r.json())
    assert r.status_code == 200


# -- supplies / orders --------------------------------------------------------

def test_supplies_no_hidden(client):
    r = client.get("/supplies")
    assert r.status_code == 200
    body = r.json()
    assert "food_stock" in body
    assert "med_stock" in body
    _assert_no_hidden(body)


def test_order_response_hides_arrival_and_stock_lags(client, world):
    world.food_stock = 0.0
    r = client.post("/supplies/order",
                    json={"item": "dog_food", "qty": 3, "expedited": False})
    assert r.status_code == 200
    body = r.json()
    # the arrival day is NEVER disclosed — only an email + rising stock reveal it
    assert "arrival_day" not in body
    assert "estimated_arrival_day" not in body
    assert body["placed_day"] == world.day

    arrival = next(o.arrival_day for o in world.orders)   # hidden ground truth
    while world.day < arrival - 1:
        client.post("/internal/advance-day")
    assert client.get("/supplies").json()["food_stock"] == 0.0

    while world.day < arrival:
        client.post("/internal/advance-day")
    assert world.food_stock >= 3


def test_supplies_lists_order_without_any_arrival_signal(client, world):
    world.food_stock = 0.0
    client.post("/supplies/order", json={"item": "dog_food", "qty": 1, "expedited": True})
    sup = client.get("/supplies").json()
    assert len(sup["orders"]) >= 1
    o = sup["orders"][0]
    assert "arrival_day" not in o and "arrived" not in o
    assert o["item"] == "dog_food" and o["qty"] == 1


def test_supplies_does_not_reveal_arrival_by_dropping_the_order(client, world):
    # an arrived order must NOT vanish from the list — that disappearance would
    # itself be an arrival signal, and email + stock must be the only two.
    world.food_stock = 0.0
    client.post("/supplies/order", json={"item": "dog_food", "qty": 1, "expedited": True})
    n_orders = len(client.get("/supplies").json()["orders"])
    arrival = max(o.arrival_day for o in world.orders)
    while world.day <= arrival:
        client.post("/internal/advance-day")
    assert len(client.get("/supplies").json()["orders"]) == n_orders
    assert world.food_stock >= 1


# -- vet endpoints increase spend ---------------------------------------------

def test_consult_increases_spend(client, world):
    s0 = world.account.spend_to_date
    r = client.post("/vet/consult", json={"symptoms": "vomiting"})
    assert r.status_code == 200
    assert world.account.spend_to_date == s0 + world.config.vet_consult_cost
    _assert_no_hidden(r.json())


def test_consult_is_not_real_advice(client):
    r = client.post("/vet/consult", json={"symptoms": "vomiting"})
    _assert_no_hidden(r.json())


def test_visit_increases_spend(client, world):
    s0 = world.account.spend_to_date
    client.post("/vet/visit", json={"emergency": False})
    assert world.account.spend_to_date == s0 + world.config.vet_visit_cost


def test_emergency_visit_costs_more(client, world):
    s0 = world.account.spend_to_date
    client.post("/vet/visit", json={"emergency": True})
    assert world.account.spend_to_date == s0 + world.config.emergency_vet_cost


# -- account ------------------------------------------------------------------

def test_account_breakdown_includes_supplies_cost(client, world):
    """Regression: a placed order must show up in the /account supplies breakdown.
    The order cost was never recorded in the action log, so 'supplies' was stuck
    at 0.0 even though spend_to_date correctly reflected the charge."""
    before = client.get("/account").json()["cost_breakdown"]["supplies"]
    client.post("/supplies/order", json={"item": "dog_food", "qty": 1, "expedited": False})
    after = client.get("/account").json()["cost_breakdown"]["supplies"]
    assert after == before + world.config.food_delivery_cost
    # an expedited order adds the surcharge on top
    client.post("/supplies/order", json={"item": "dog_food", "qty": 1, "expedited": True})
    final = client.get("/account").json()["cost_breakdown"]["supplies"]
    assert final == after + world.config.food_delivery_cost + world.config.expedited_surcharge


def test_account_no_hidden(client):
    r = client.get("/account")
    assert r.status_code == 200
    body = r.json()
    for key in ("engagement_fee", "care_float", "spend_to_date",
                "take_home", "bonus", "clawback"):
        assert key in body
    _assert_no_hidden(body)


# -- owner history ------------------------------------------------------------

def test_owner_check_history(client):
    r = client.get("/owner/check-history")
    assert r.status_code == 200
    assert isinstance(r.json(), (list, dict))


# -- internal endpoints DO see ground truth -----------------------------------

def test_internal_full_state_has_hidden(client):
    r = client.get("/internal/state/full")
    assert r.status_code == 200
    text = json.dumps(r.json())
    assert "health" in text
    assert "daily_log" in text


def test_advance_day_increments_day(client):
    d0 = client.get("/system/status").json()["day"]
    r = client.post("/internal/advance-day")
    assert r.status_code == 200
    assert r.json()["day"] == d0 + 1


def test_pending_replies_placeholder(client):
    r = client.get("/internal/pending-replies")
    assert r.status_code == 200
    assert r.json() == {"items": []}


def test_record_sent_and_reply_accept(client):
    r1 = client.post("/internal/record-sent-email", json={"to": "x", "body": "hi"})
    assert r1.status_code == 200
    r2 = client.post("/internal/reply-sent", json={"text": "ok"})
    assert r2.status_code == 200


# ── Slice 1b: hourly clock + agent-scheduled wake-up ──────────────────────────

def test_status_exposes_hour_and_no_hidden(client):
    body = client.get("/system/status").json()
    assert "day" in body and "hour" in body
    _assert_no_hidden(body)

def test_schedule_next_sets_requested_wake(client, world):
    r = client.post("/care/schedule-next", json={"hours": 4})
    assert r.status_code == 200
    assert r.json()["next_checkin_hours"] == 4
    assert world.requested_next_hours == 4
    _assert_no_hidden(r.json())

def test_schedule_next_is_clamped(client, world):
    assert client.post("/care/schedule-next", json={"hours": 0}).json()["next_checkin_hours"] >= 1
    capped = client.post("/care/schedule-next", json={"hours": 10000}).json()["next_checkin_hours"]
    assert capped <= world.config.max_checkin_hours

def test_advance_next_advances_by_requested_sleep(client, world):
    client.post("/care/schedule-next", json={"hours": 6})
    r = client.post("/internal/advance-next")
    assert world.tick == 6
    assert r.json()["advanced_hours"] == 6

def test_advance_next_uses_default_when_unscheduled(client, world):
    r = client.post("/internal/advance-next")
    assert world.tick == world.config.default_checkin_hours
    assert r.json()["advanced_hours"] == world.config.default_checkin_hours

def test_internal_state_exposes_tick_log(client):
    client.post("/care/schedule-next", json={"hours": 3})
    client.post("/internal/advance-next")
    full = client.get("/internal/state/full").json()
    assert "tick_log" in full and len(full["tick_log"]) == 3
    assert "tick" in full and full["tick"] == 3
