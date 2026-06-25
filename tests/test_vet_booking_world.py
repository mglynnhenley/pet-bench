"""Async vet-visit arrival in the world tick (2026-06-25).

An emailed appointment doesn't teleport a vet into the home — the booked vet
arrives during the next check-in gap (`request_vet_visit` registers a pending
visit; the hourly tick lands it at its ETA). On arrival the visit becomes the
same physical, observed event the old console action produced: a vet shows on
the camera (`vet_present_hours`), an emergency stabilises a critical dog, the
cost is charged, and the visit goes on record (`vet_visits_landed`) so the vet's
later replies are grounded. Because help is no longer instant, a critical dog
can decline — even die — before the vet gets there.

Determinism: incident/camera probs 0; state driven directly.
"""

import pytest

from petcare.config import Config
from petcare.world import new_world


def _world(**cfg):
    base = dict(illness_onset_prob=0.0, camera_offline_prob=0.0,
                camera_misread_prob=0.0, camera_obscured_prob=0.0)
    base.update(cfg)
    return new_world(Config(**base), seed=1)


# ── the booked vet arrives after its ETA, not instantly ─────────────────────────

def test_visit_does_not_land_before_its_eta():
    w = _world()
    w.request_vet_visit(emergency=False, eta_hours=4, cost=80.0)
    w.advance(3)                       # 3h < 4h ETA
    assert w.vet_present_hours == 0
    assert w.vet_visits_landed == []


def test_visit_lands_at_its_eta_and_shows_on_camera():
    w = _world()
    w.request_vet_visit(emergency=False, eta_hours=4, cost=80.0)
    w.advance(4)
    assert w.vet_present_hours > 0     # a vet is now on site
    assert w.tick_log[-1]["visible_behaviour"] == "vet_visit"
    assert len(w.vet_visits_landed) == 1
    rec = w.vet_visits_landed[0]
    assert rec["kind"] == "visit" and rec["emergency"] is False
    assert rec["cost"] == 80.0


def test_arrival_charges_the_account_once():
    w = _world()
    start = w.account.spend_to_date
    w.request_vet_visit(emergency=False, eta_hours=2, cost=80.0)
    w.advance(24)                      # well past arrival
    assert w.account.spend_to_date == start + 80.0
    assert len(w.vet_visits_landed) == 1   # lands exactly once


# ── routine vs emergency on arrival ─────────────────────────────────────────────

def test_routine_visit_on_arrival_does_not_treat():
    w = _world()
    w.set_illness("gi_upset", 3)
    w.dog.health = 50.0
    w.request_vet_visit(emergency=False, eta_hours=2, cost=80.0)
    w.advance(2)
    assert w.dog.illness_severity == 3      # assessment only


def test_emergency_visit_on_arrival_stabilises():
    w = _world()
    w.set_illness("gi_upset", 3)
    w.dog.health = 5.0                       # critical
    w.request_vet_visit(emergency=True, eta_hours=2, cost=200.0)
    w.advance(2)
    assert w.dog.health > w.config.critical_health_threshold
    assert w.dog.illness_severity <= 1


# ── async means help can come too late ──────────────────────────────────────────

def test_dog_can_die_before_the_booked_vet_arrives():
    w = _world(death_after_critical_hours=3)
    w.set_illness("gi_upset", 3)
    w.dog.health = 4.0                       # already critical
    w.request_vet_visit(emergency=True, eta_hours=6, cost=200.0)
    w.advance(6)                             # dies at hour 3, vet arrives hour 6
    assert w.dog.alive is False
    assert w.death_event is not None


def test_same_hour_emergency_arrival_can_save_the_dog():
    w = _world(death_after_critical_hours=4)
    w.set_illness("gi_upset", 3)
    w.dog.health = 4.0
    w.request_vet_visit(emergency=True, eta_hours=3, cost=200.0)
    w.advance(3)                             # stabilises before the 4th critical hour
    assert w.dog.alive is True
    assert w.dog.health > w.config.critical_health_threshold


# ── soonest-arriving booking wins when two are outstanding ──────────────────────

def test_second_sooner_booking_lands_first():
    w = _world()
    w.request_vet_visit(emergency=False, eta_hours=6, cost=80.0)
    w.request_vet_visit(emergency=True, eta_hours=2, cost=200.0)   # sooner
    w.advance(2)
    assert len(w.vet_visits_landed) == 1
    assert w.vet_visits_landed[0]["emergency"] is True
