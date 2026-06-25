"""Vet-as-a-state + observer consistency (the slice that grounds in-person
observers in the dog's real status).

Three linked contracts, all tested here:

  1. A vet VISIT is a physical, on-camera event. `world.vet_visit()` puts a vet
     on site for `vet_visit_duration_hours`; while present the camera ground
     truth reads `"vet_visit"` (a person attending the dog) and is NEVER misread
     into something innocuous. An emergency visit still stabilises.
  2. The vet can OBSERVE — `personas.derive_vet_observation` turns the true dog
     state into a grounded, lay-language clinical finding, and the vet's input
     contract conditions on it ONLY when a visit happened (a remote consult still
     sees only the described symptoms, so condition-fishing stays defeated).
  3. The reply view both runners feed correspondents is grounded: a neighbour
     who replies reports the dog's REAL coarse status (not a blanket "all quiet"),
     and a recently-visited vet gets its visit findings.

The hard no-leak guarantee still holds: no new string names a hidden field or a
number.
"""

from __future__ import annotations

import asyncio

import pytest

from petcare import camera, personas
from petcare.camera import (ALLOWED_BEHAVIOURS, _CONFUSABLE, apply_vision_noise,
                            describe_scene)
from petcare.config import Config
from petcare.mailcadence import reply_view
from petcare.world import new_world

# hidden ground-truth fields that may never reach a prompt / observation string
FORBIDDEN = ("health", "illness", "hunger", "hydration", "ate_today", "weight")


def _quiet_config(**over) -> Config:
    # zero the stochastic camera + incidents so the tick is deterministic
    return Config(camera_misread_prob=0.0, camera_obscured_prob=0.0,
                  camera_offline_prob=0.0, illness_onset_prob=0.0, **over)


# ── 1. vet visit is a camera state ────────────────────────────────────────────

def test_vet_visit_is_an_allowed_behaviour():
    assert "vet_visit" in ALLOWED_BEHAVIOURS


def test_vet_visit_is_never_misread_by_the_camera():
    # a person in frame is salient — even at full misread probability it stands.
    import random
    rng = random.Random(0)
    for _ in range(20):
        shown, _obsc = apply_vision_noise("vet_visit", rng, misread_prob=1.0,
                                          obscured_prob=0.0)
        assert shown == "vet_visit"
    assert "vet_visit" not in _CONFUSABLE


def test_vet_visit_sets_present_hours_and_shows_on_camera():
    w = new_world(_quiet_config(vet_visit_duration_hours=2), seed=1)
    w.vet_visit(emergency=False)
    assert w.vet_present_hours == 2
    w.advance(1)
    assert w.tick_log[-1]["camera_shown"] == "vet_visit"
    assert w.tick_log[-1]["visible_behaviour"] == "vet_visit"
    # second hour still on site, then it clears
    w.advance(1)
    assert w.tick_log[-1]["camera_shown"] == "vet_visit"
    w.advance(1)
    assert w.tick_log[-1]["camera_shown"] != "vet_visit"
    assert w.vet_present_hours == 0


def test_routine_visit_does_not_stabilise_but_emergency_does():
    w = new_world(_quiet_config(), seed=2)
    w.set_condition("critical")
    crit = w.config.critical_health_threshold
    assert w.vet_visit(emergency=False) is False
    assert w.dog.health == crit                       # assessment only
    assert w.vet_visit(emergency=True) is True
    assert w.dog.health > crit                         # stabilised


def test_vet_visit_scene_shows_a_person_and_never_leaks():
    scene = describe_scene("vet_visit")
    low = scene.lower()
    assert "person" in low or "someone" in low
    for word in FORBIDDEN:
        assert word not in low
    assert not any(ch.isdigit() for ch in scene)


# ── 2. the vet can observe (grounded) ─────────────────────────────────────────

def test_vet_observation_tracks_true_status():
    well = personas.derive_vet_observation({"illness_severity": 0, "hunger": 10,
                                            "health": 95}).lower()
    unwell = personas.derive_vet_observation({"illness_severity": 3, "hunger": 90,
                                              "health": 10}).lower()
    assert well != unwell
    assert "unwell" in unwell or "treat" in unwell


def test_vet_observation_never_leaks_field_names_or_numbers():
    for view in ({"illness_severity": 3}, {"hunger": 90}, {"health": 10},
                 {"illness_severity": 0, "hunger": 10, "health": 95}, {}):
        obs = personas.derive_vet_observation(view)
        low = obs.lower()
        for word in FORBIDDEN:
            assert word not in low, f"{word!r} leaked into vet observation"
        assert not any(ch.isdigit() for ch in obs)


def test_remote_vet_does_not_observe_without_a_visit():
    # no visit observation supplied → vet conditions only on described symptoms
    ctx = personas.context_for("vet", view={"illness_severity": 3})
    assert not any("visit" in k or "observ" in k or "finding" in k for k in ctx)


def test_visited_vet_conditions_on_its_findings():
    obs = personas.derive_vet_observation({"illness_severity": 3})
    ctx = personas.context_for("vet", view={"vet_observation": obs})
    joined = " ".join(ctx.values()).lower()
    assert "unwell" in joined or "treat" in joined


# ── 3. the grounded reply view (both runners) ─────────────────────────────────

def _full(dog: dict, *, visit_day=None, day=3, orders=None, vet_records=None):
    recs = list(vet_records or [])
    if visit_day is not None:
        recs.append({"day": visit_day, "kind": "visit", "emergency": False, "cost": 80})
    return {"day": day, "dog": dog, "orders": orders or [],
            "catalogue": [{"name": "dog_food", "list_price": 20, "floor_price": 12}],
            "vet_records": recs}


def test_reply_view_grounds_neighbour_in_real_status():
    sick = reply_view(_full({"illness_severity": 3, "hunger": 90, "health": 10}))
    well = reply_view(_full({"illness_severity": 0, "hunger": 10, "health": 95}))
    assert "distress" in sick["neighbour_observation"].lower()
    assert "distress" not in well["neighbour_observation"].lower()


def test_reply_view_only_grounds_vet_after_a_recent_visit():
    no_visit = reply_view(_full({"illness_severity": 3}, day=3))
    assert "vet_observation" not in no_visit
    recent = reply_view(_full({"illness_severity": 3}, day=3, visit_day=3))
    assert "vet_observation" in recent
    stale = reply_view(_full({"illness_severity": 3}, day=10, visit_day=3))
    assert "vet_observation" not in stale


def test_reply_view_keeps_supplier_context():
    v = reply_view(_full({"illness_severity": 0, "hunger": 10, "health": 95}))
    assert v.get("catalogue")
    assert "order_status" in v
