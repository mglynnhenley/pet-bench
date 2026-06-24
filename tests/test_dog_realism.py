"""Dog realism tuning: distinct water restore, tapered sick appetite, weight regain.

These tweaks make the dog's response to care more lifelike without changing the
three core gaps (dispense≠eat, order≠arrive, dose≠cure). All knobs live in Config.
"""

import pytest

from petcare.config import Config
from petcare.world import new_world


@pytest.fixture
def world():
    return new_world(Config(illness_onset_prob=0.0, camera_offline_prob=0.0,
                            camera_misread_prob=0.0, camera_obscured_prob=0.0), seed=1)


# ── water has its own restore amount (no longer reusing eat_amount) ───────────

def test_drink_restores_hydration_by_water_amount(world):
    world.dog.hydration = 20.0
    world.give_water()
    world.advance(1)
    assert world.dog.hydration == pytest.approx(20.0 + world.config.water_amount)


# ── appetite tapers with illness instead of being all-or-nothing ─────────────

def test_mildly_ill_dog_eats_but_less_than_a_healthy_one():
    cfg = Config(illness_onset_prob=0.0, camera_offline_prob=0.0)
    healthy = new_world(cfg, seed=1)
    sick = new_world(cfg, seed=1)
    sick.set_illness("gi_upset", 1)            # mild: off its food but still picks at it
    for w in (healthy, sick):
        w.dog.hunger = 60.0
        w.dispense_food()
        w.advance(1)
    assert healthy.dog.ate_today is True
    assert sick.dog.ate_today is True          # a mildly-ill dog still eats something
    assert sick.dog.hunger > healthy.dog.hunger  # ...but less, so it stays hungrier


def test_more_than_mild_illness_still_refuses_food_entirely(world):
    world.set_illness("gi_upset", 2)
    world.dispense_food()
    world.advance_day()
    assert world.dog.ate_today is False


# ── weight creeps back toward healthy when well-fed and well ──────────────────

def test_underweight_well_fed_dog_regains_weight(world):
    world.dog.weight = 18.0
    world.dog.hunger = 10.0
    world.dispense_food()                      # one portion → ate_today holds for the day
    world.advance(24)
    assert world.dog.weight > 18.0
    assert world.dog.weight <= world.config.healthy_weight


def test_starving_dog_does_not_regain_weight(world):
    w0 = world.dog.weight
    for _ in range(8):                         # never fed → no regain, only loss
        world.advance_day()
    assert world.dog.weight < w0
