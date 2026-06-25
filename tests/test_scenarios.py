"""Scenario authoring (slice: scenarios).

Each named scenario maps to a tuning ``Config`` (via ``build_config``) and an
optional seeded starting state (via ``_seed_scenario_state``). Existing names
(baseline / thin_margin / illness_episode) must keep working; unknown names fall
back to baseline (never raise). These tests are written first (RED).
"""

from dataclasses import replace

from petcare.camera import derive_visible_behaviour
from petcare.config import Config
from petcare.server import build_config, build_world


# ── existing names unchanged ─────────────────────────────────────────────────

def test_baseline_is_default_config():
    assert build_config("baseline") == Config()


def test_unknown_falls_back_to_baseline():
    assert build_config("does-not-exist") == Config()
    assert build_config("noisy_cammera_typo") == build_config("baseline")


def test_thin_margin_unchanged():
    assert build_config("thin_margin") == replace(Config(), care_float=75.0)


def test_illness_episode_unchanged():
    assert build_config("illness_episode") == replace(Config(), illness_onset_prob=0.6)


def test_illness_episode_seeds_sick_dog():
    w = build_world("illness_episode", seed=3)
    assert w.dog.illness_type is not None
    assert w.dog.illness_severity >= 1
    assert w.dog.illness_load > 0


def test_baseline_seeds_healthy_dog():
    w = build_world("baseline", seed=3)
    assert w.dog.illness_type is None
    assert w.dog.illness_severity == 0


# ── new scenarios: config overrides ──────────────────────────────────────────

def test_delivery_trouble_raises_delay_prob():
    cfg = build_config("delivery_trouble")
    assert cfg.delivery_delay_prob >= 0.8
    # other knobs unaffected (only delivery knobs differ from baseline)
    assert cfg.care_float == Config().care_float


def test_noisy_camera_raises_camera_noise():
    cfg = build_config("noisy_camera")
    assert cfg.camera_misread_prob >= 0.4
    assert cfg.camera_obscured_prob >= 0.3


def test_demanding_owner_raises_owner_pressure():
    cfg = build_config("demanding_owner")
    cadence = cfg.cadence
    base = Config().cadence
    assert cadence["owner_nudge_prob_per_day"] > base["owner_nudge_prob_per_day"]
    assert cadence["owner_silence_hours"] < base["owner_silence_hours"]


# ── new scenarios: seeded state ──────────────────────────────────────────────

def test_limp_injury_shows_limping_behaviour():
    w = build_world("limp_injury", seed=5)
    assert w.dog.illness_type == "limp"
    # severity high enough that the camera-visible behaviour is 'limping'
    assert derive_visible_behaviour(w.dog) == "limping"


def test_lethargy_creep_starts_mildly_lethargic():
    w = build_world("lethargy_creep", seed=5)
    assert w.dog.illness_type == "lethargy"
    assert w.dog.illness_severity == 1
    assert derive_visible_behaviour(w.dog) == "lethargic"


def test_config_only_scenarios_leave_dog_healthy():
    for name in ("delivery_trouble", "noisy_camera", "demanding_owner"):
        w = build_world(name, seed=5)
        assert w.dog.illness_type is None, name
        assert w.dog.illness_severity == 0, name


def test_unknown_scenario_builds_baseline_world():
    w = build_world("does-not-exist", seed=5)
    assert w.config == Config()
    assert w.dog.illness_type is None
