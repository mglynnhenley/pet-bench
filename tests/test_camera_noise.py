"""Imperfect-vision camera noise: ground truth vs. what the camera reports.

The camera runs an imperfect AI vision read over the footage. It can (a) misread
one appearance as a similar-looking one and (b) get a partial/obscured view. The
true appearance (`visible_behaviour`) is always recorded as ground truth; the
noise only perturbs what the camera *shows* (`camera_shown` / `camera_obscured`).
These tests are the contract for that separation.
"""

import asyncio
import random

from petcare.config import Config
from petcare.world import new_world
from petcare.camera import (
    ALLOWED_BEHAVIOURS,
    _CONFUSABLE,
    _build_prompt,
    apply_vision_noise,
    build_windows,
    caption,
    describe_scene,
)


class _ExplodingRng:
    """An rng that raises if touched — proves apply_vision_noise makes no draws."""

    def random(self):
        raise AssertionError("rng.random() must not be called when probs are 0")

    def randrange(self, n):
        raise AssertionError("rng.randrange() must not be called when probs are 0")


# ── apply_vision_noise ────────────────────────────────────────────────────────

def test_noise_is_strict_noop_when_probs_zero():
    shown, obscured = apply_vision_noise("resting", _ExplodingRng(),
                                         misread_prob=0.0, obscured_prob=0.0)
    assert shown == "resting"
    assert obscured is False


def test_misread_prob_one_yields_a_plausible_different_label():
    rng = random.Random(0)
    shown, _ = apply_vision_noise("resting", rng, misread_prob=1.0, obscured_prob=0.0)
    assert shown != "resting"
    assert shown in _CONFUSABLE["resting"]
    assert shown in ALLOWED_BEHAVIOURS


def test_obscured_prob_one_marks_obscured():
    _, obscured = apply_vision_noise("pacing", random.Random(0),
                                     misread_prob=0.0, obscured_prob=1.0)
    assert obscured is True


def test_shown_label_is_always_allowed_for_every_true_label():
    rng = random.Random(1)
    for label in ALLOWED_BEHAVIOURS:
        for _ in range(20):
            shown, _ = apply_vision_noise(label, rng, misread_prob=1.0, obscured_prob=0.5)
            assert shown in ALLOWED_BEHAVIOURS


# ── world records ground truth AND the reported view ──────────────────────────

def test_world_records_camera_shown_and_obscured_fields():
    w = new_world(Config(illness_onset_prob=0.0, camera_offline_prob=0.0), seed=1)
    w.advance(3)
    for row in w.tick_log:
        assert "visible_behaviour" in row   # ground truth
        assert "camera_shown" in row        # what the camera reported
        assert "camera_obscured" in row
        assert "camera_misread" in row
        assert row["camera_shown"] in ALLOWED_BEHAVIOURS


def test_ground_truth_is_preserved_when_noise_disabled():
    # With both noise knobs off, the reported view equals the truth exactly.
    cfg = Config(illness_onset_prob=0.0, camera_offline_prob=0.0,
                 camera_misread_prob=0.0, camera_obscured_prob=0.0)
    w = new_world(cfg, seed=3)
    w.advance(48)
    for row in w.tick_log:
        assert row["camera_shown"] == row["visible_behaviour"]
        assert row["camera_misread"] is False
        assert row["camera_obscured"] is False


def test_noise_makes_camera_diverge_from_truth_but_truth_still_tracked():
    # Force misreads: some hours the camera shows the wrong label, yet the
    # hidden ground truth keeps tracking the real appearance.
    cfg = Config(illness_onset_prob=0.0, camera_offline_prob=0.0,
                 camera_misread_prob=1.0, camera_obscured_prob=0.0)
    w = new_world(cfg, seed=2)
    w.set_illness("limp", 3)   # a true, persistent visible symptom (limping)
    w.advance(24)
    assert all(r["visible_behaviour"] == "limping" for r in w.tick_log)  # truth held
    assert any(r["camera_misread"] for r in w.tick_log)                  # camera lied sometimes
    assert any(r["camera_shown"] != "limping" for r in w.tick_log)


# ── obscured captions ─────────────────────────────────────────────────────────

def test_obscured_scene_is_tentative_and_prompt_stays_leak_free():
    # An obscured window's SCENE reads as an unclear, partial view...
    scene = describe_scene("limping", obscured=True)
    assert any(w in scene.lower()
               for w in ("blocked", "hard to make out", "in the way", "hard to read"))
    # ...and the prompt built from it still leaks no hidden words or digits.
    prompt = _build_prompt(scene)
    lowered = prompt.lower()
    for word in ("health", "illness", "hunger", "hydration"):
        assert word not in lowered
    assert not any(ch.isdigit() for ch in prompt)


def test_scene_is_deterministic_without_rng_but_varies_with_one():
    # No rng → stable phrasing (tests stay deterministic).
    assert describe_scene("pacing") == describe_scene("pacing")
    # With an rng, the camera draws from several phrasings of the same scene, so
    # across many windows it does not always say the exact same thing.
    rng = random.Random(0)
    seen = {describe_scene("pacing", rng=rng) for _ in range(40)}
    assert len(seen) > 1
    assert all("dog" in s for s in seen)   # still describing the same dog


def test_obscured_window_still_renders_one_timestamped_line():
    class _Fake:
        async def generate(self, prompt):
            return "the view is partly blocked but a dog seems to be present"

    windows = build_windows(["resting"], [False], obscured=[True])
    assert windows[0].obscured is True
    transcript = asyncio.run(caption(_Fake(), windows))
    assert transcript.startswith("[06:00]")
