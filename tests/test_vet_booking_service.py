"""Console surface for the email-booked vet (2026-06-25).

The agent no longer has a `POST /vet/visit` console action — an in-person visit
is booked by emailing the vet, and the runner registers the booking through the
internal `POST /internal/book-vet-visit`. When the booked vet arrives during a
later advance, the visit is drained onto the vet record so `/account` and the
vet persona's "recently visited" grounding stay correct. The remote
`POST /vet/consult` console action is unchanged.
"""

import pytest
from fastapi.testclient import TestClient

from petcare.config import Config
from petcare.world import new_world
from petcare.service import create_app


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


# ── the console visit action is gone ────────────────────────────────────────────

def test_vet_visit_console_action_is_removed(client):
    r = client.post("/vet/visit", json={"emergency": False})
    assert r.status_code == 404


def test_consult_still_works(client):
    r = client.post("/vet/consult", json={"symptoms": "off his food"})
    assert r.json()["ok"] is True


# ── booking via the internal endpoint ───────────────────────────────────────────

def test_booking_registers_a_pending_visit(client, world):
    r = client.post("/internal/book-vet-visit", json={"emergency": False})
    body = r.json()
    assert body["booked"] is True
    assert body["eta_hours"] == world.config.vet_visit_eta_hours
    assert body["cost"] == world.config.vet_visit_cost
    assert world._pending_vet_visit is not None


def test_emergency_booking_uses_the_emergency_eta_and_cost(client, world):
    r = client.post("/internal/book-vet-visit", json={"emergency": True})
    body = r.json()
    assert body["booked"] is True
    assert body["eta_hours"] == world.config.emergency_vet_eta_hours
    assert body["cost"] == world.config.emergency_vet_cost


# ── arrival lands on the record + the account ───────────────────────────────────

def test_arrival_during_advance_lands_on_the_vet_record(client, world):
    client.post("/internal/book-vet-visit", json={"emergency": False})
    client.post("/care/schedule-next", json={"hours": world.config.vet_visit_eta_hours})
    client.post("/internal/advance-next")
    full = client.get("/internal/state/full").json()
    visits = [r for r in full["vet_records"] if r["kind"] == "visit"]
    assert len(visits) == 1
    # cost shows up in the account breakdown
    assert client.get("/account").json()["cost_breakdown"]["vet"] >= world.config.vet_visit_cost


def test_visit_not_recorded_before_it_arrives(client, world):
    client.post("/internal/book-vet-visit", json={"emergency": False})
    client.post("/care/schedule-next", json={"hours": 1})   # 1h < ETA
    client.post("/internal/advance-next")
    full = client.get("/internal/state/full").json()
    assert [r for r in full["vet_records"] if r["kind"] == "visit"] == []
