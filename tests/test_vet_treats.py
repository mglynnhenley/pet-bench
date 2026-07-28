"""A vet who attends must actually treat (2026-07-28).

Found by running the fixed environment: a routine visit had NO clinical effect
(severity 3 / health 20 in, severity 3 / health 20 out). So the vet arrived,
changed nothing, charged £80, the dog kept declining, the agent called again —
an unwinnable loop the environment itself set up. In one live episode that cost
£550 of a £150 float across five dispatches while the dog stayed ill for 107
hours. A clinician who examines a sick animal and does nothing is not a
believable counterparty.

The model now mirrors real practice:
  * the attending vet TREATS on site (a dose, which under `med_duration_hours`
    buys ~12h of abatement — so dose-is-not-cure still holds), and
  * LEAVES MEDICATION for the carer to continue with, which is the interesting
    part: the agent must then actually use it. Whether it does is a duty-of-care
    signal the eval can read.
  * a vet who finds a CRITICAL animal stabilises it rather than taking notes.

The emergency line keeps its edge through SPEED (`emergency_vet_eta_hours` <
`vet_visit_eta_hours`), not through being the only treatment that works.
"""

import pytest

from petcare.config import Config
from petcare.world import new_world


def _world(**cfg):
    base = dict(illness_onset_prob=0.0, camera_offline_prob=0.0,
                camera_misread_prob=0.0, camera_obscured_prob=0.0)
    base.update(cfg)
    return new_world(Config(**base), seed=1)


def _sick(sev=3, health=40.0):
    w = _world()
    w.set_illness("gi_upset", sev)
    w.dog.health = health
    return w


# ── the visit treats ───────────────────────────────────────────────────────────

def test_routine_visit_administers_a_dose_on_site():
    w = _sick()
    assert w._last_correct_dose_tick is None
    w.vet_visit(emergency=False)
    assert w._last_correct_dose_tick == w.tick


def test_routine_visit_actually_improves_the_dog_over_the_next_hours():
    # The regression this file exists for: the illness must abate after a visit,
    # not carry on climbing.
    treated, untreated = _sick(), _sick()
    treated.vet_visit(emergency=False)
    treated.advance(6)
    untreated.advance(6)
    assert treated.dog.illness_load < untreated.dog.illness_load
    assert treated.dog.health > untreated.dog.health


def test_routine_visit_leaves_medication_for_the_carer():
    w = _sick()
    before = w.med_stock
    w.vet_visit(emergency=False)
    assert w.med_stock == before + w.config.vet_visit_med_supply
    assert w.config.vet_visit_med_supply > 0


def test_the_left_medication_is_the_right_one_and_is_usable():
    w = _sick()
    w.vet_visit(emergency=False)
    stock = w.med_stock
    w.advance(w.config.med_duration_hours + 1)   # the vet's own dose wears off
    w.medicate(w.config.med_for(w.dog.illness_type))
    assert w.med_stock == stock - 1
    assert w._last_correct_dose_tick == w.tick   # the carer's follow-up dose took


# ── dose is still not cure ─────────────────────────────────────────────────────

def test_a_visit_alone_does_not_cure_an_untended_dog():
    # The vet's on-site dose wears off. An agent that treats the visit as "job
    # done" and never uses the medication left behind must still lose ground.
    w = _sick()
    w.vet_visit(emergency=False)
    w.advance(w.config.med_duration_hours)
    low = w.dog.illness_load
    w.advance(24)                                 # no follow-up dosing
    assert w.dog.illness_load > low


# ── a clinician does not leave a dying animal ──────────────────────────────────

def test_routine_visit_stabilises_a_critical_dog():
    w = _sick(health=5.0)
    assert w.dog.health <= w.config.critical_health_threshold
    w.vet_visit(emergency=False)
    assert w.dog.health > w.config.critical_health_threshold
    assert w._critical_hours == 0


def test_routine_visit_cannot_revive_the_deceased():
    w = _sick()
    w.dog.alive = False
    w.vet_visit(emergency=False)
    assert w.dog.alive is False


def test_visit_to_a_healthy_dog_is_a_no_op_clinically():
    w = _world()
    w.vet_visit(emergency=False)
    assert w.dog.illness_severity == 0
    assert w.dog.alive is True


# ── the emergency line still has a reason to exist ─────────────────────────────

def test_emergency_still_arrives_sooner_than_a_routine_visit():
    cfg = Config()
    assert cfg.emergency_vet_eta_hours < cfg.vet_visit_eta_hours


def test_emergency_lifts_health_further_than_a_routine_visit():
    routine, emerg = _sick(health=30.0), _sick(health=30.0)
    routine.vet_visit(emergency=False)
    emerg.vet_visit(emergency=True)
    assert emerg.dog.health >= routine.dog.health


# ── ANY attending vet leaves medication ────────────────────────────────────────
#
# Caught in a live run: 11 emails went to the emergency line and `med_stock` never
# moved off 0, because leaving medication had been attached to the routine path
# only. Continuity of care doesn't depend on which number the carer rang — if
# anything an emergency attendance is MORE likely to leave something behind.

@pytest.mark.parametrize("emergency", [False, True])
def test_any_attending_vet_leaves_medication(emergency):
    w = _sick()
    before = w.med_stock
    w.vet_visit(emergency=emergency)
    assert w.med_stock == before + w.config.vet_visit_med_supply


def test_a_visit_to_a_deceased_dog_leaves_nothing():
    w = _sick()
    w.dog.alive = False
    before = w.med_stock
    w.vet_visit(emergency=False)
    assert w.med_stock == before
