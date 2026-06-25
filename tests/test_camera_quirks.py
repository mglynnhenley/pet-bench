"""Camera realism pass (user request 2026-06-24):
  • the captioner prompt is reframed to read like a TERSE auto-caption (clipped,
    variable length, lower confidence, may mention camera artifacts) instead of
    smooth novelist prose — while keeping the observational "describe, don't
    diagnose" guard and the no-leak contract;
  • describe_scene draws from MORE interchangeable phrasings per appearance and
    may layer rng-sampled "camera quirks" (low light, grain, motion zone, edge
    of frame) so consecutive windows don't read identically.

Hard contracts re-asserted here:
  1. NO LEAK — no phrasing, quirk, or prompt contains a hidden-state word, an
     illness name, or any digit.
  2. DETERMINISM — describe_scene(rng=None) is fully stable and applies NO
     quirks; supplying an rng produces variation.
"""

import random
import re

from petcare.camera import (
    ALLOWED_BEHAVIOURS,
    _SCENE_PHRASINGS,
    _build_prompt,
    describe_scene,
)

# Whole-word forbidden tokens: hidden-state field names + illness names. Matched
# on word boundaries so innocent appearance words ("water", "lopsided",
# "auto-generates") don't false-positive, while the actual field/illness tokens
# ("ate", "weight", "limp", "lethargy") are still banned as standalone words.
_FORBIDDEN_WORDS = (
    "health", "illness", "hunger", "hydration", "ate", "weight",
    "gi_upset", "lethargy", "limp",
)

# DESCRIBE, DON'T CONCLUDE (user request 2026-06-25): the camera reports what is
# physically on screen — posture, position, movement, stillness — and leaves the
# judgement (is the dog OK? comfortable? in trouble?) to the agent. So the scene
# phrasings must carry no welfare/mood/affect *conclusion* and no asserted vital
# sign. "breathing" is banned because a still dog on a grainy feed can't be shown
# to be breathing — asserting it fabricates a sign of life (the dead-dog bug).
# Note: these are checked on the SCENE prose only, not on _build_prompt's
# instruction text, which must name these categories in order to forbid them.
_CONCLUSION_WORDS = (
    "calm", "relaxed", "relaxing", "settled", "unsettled", "comfortable", "comfy",
    "content", "contented", "peaceful", "cosy", "cozy", "serene", "at ease", "ease",
    "easy", "easily", "dozing", "doze", "asleep", "sleeping", "sleepy", "napping",
    "breathing", "breathe", "breaths", "happy", "fine", "unbothered", "untroubled",
    "distressed", "distress", "agitated", "anxious", "nervous", "weary", "exhausted",
    "sluggish", "sluggishly", "listless", "dull", "lifeless", "restless", "fidgety",
    "fidgeting", "unhurried", "tired", "comfortably", "contentedly", "peacefully",
)


def _assert_leak_free(text: str):
    low = text.lower()
    for word in _FORBIDDEN_WORDS:
        assert not re.search(rf"\b{re.escape(word)}\b", low), \
            f"forbidden word {word!r} in: {text!r}"
    assert not any(ch.isdigit() for ch in text), f"digit leaked in: {text!r}"


def _assert_no_conclusions(text: str):
    low = text.lower()
    for word in _CONCLUSION_WORDS:
        assert not re.search(rf"\b{re.escape(word)}\b", low), \
            f"conclusion/affect word {word!r} (camera should describe, not judge) in: {text!r}"


# ── more phrasings, still appearance-only ─────────────────────────────────────

def test_each_behaviour_has_several_phrasings():
    for label in ALLOWED_BEHAVIOURS:
        assert len(_SCENE_PHRASINGS[label]) >= 5


def test_every_phrasing_is_leak_free():
    for label, phrasings in _SCENE_PHRASINGS.items():
        for p in phrasings:
            _assert_leak_free(p)


# ── describe, don't conclude: scene prose carries no welfare/mood judgement ───

def test_every_phrasing_makes_no_welfare_conclusion():
    for label, phrasings in _SCENE_PHRASINGS.items():
        for p in phrasings:
            _assert_no_conclusions(p)


def test_rng_driven_scenes_make_no_welfare_conclusion():
    rng = random.Random(99)
    for label in ALLOWED_BEHAVIOURS:
        for _ in range(80):
            scene = describe_scene(label, rng=rng, obscured=bool(rng.getrandbits(1)))
            _assert_no_conclusions(scene)


# ── determinism with rng=None (no quirks, first phrasing) ─────────────────────

def test_no_rng_is_deterministic_and_quirk_free_for_every_behaviour():
    for label in ALLOWED_BEHAVIOURS:
        a = describe_scene(label)
        b = describe_scene(label)
        assert a == b
        # rng=None must pick the FIRST phrasing verbatim — no quirk appended
        assert a == _SCENE_PHRASINGS[label][0]


def test_obscured_no_rng_is_deterministic():
    for label in ALLOWED_BEHAVIOURS:
        assert describe_scene(label, obscured=True) == describe_scene(label, obscured=True)


# ── variation + leak-safety when an rng is supplied ───────────────────────────

def test_rng_produces_variation_for_every_behaviour():
    for label in ALLOWED_BEHAVIOURS:
        rng = random.Random(0)
        seen = {describe_scene(label, rng=rng) for _ in range(60)}
        assert len(seen) > 1, f"{label} did not vary with an rng"


def test_rng_driven_scenes_stay_leak_free():
    rng = random.Random(123)
    for label in ALLOWED_BEHAVIOURS:
        for _ in range(80):
            scene = describe_scene(label, rng=rng, obscured=bool(rng.getrandbits(1)))
            _assert_leak_free(scene)
            assert scene  # never empty
            assert "\n" not in scene  # single logical line


def test_rng_driven_prompts_stay_leak_free():
    rng = random.Random(7)
    for label in ALLOWED_BEHAVIOURS:
        for _ in range(40):
            scene = describe_scene(label, rng=rng)
            _assert_leak_free(_build_prompt(scene))


# ── reframed prompt: terse auto-caption, still observational + leak-free ───────

def test_prompt_reads_like_a_terse_auto_caption():
    prompt = _build_prompt(describe_scene("resting")).lower()
    # terse / clipped framing, not "vivid novelist" framing
    assert "terse" in prompt or "clipped" in prompt or "brief" in prompt
    # keeps the observational guard
    assert "diagnose" in prompt
    # may reference camera artifacts
    assert any(w in prompt for w in
               ("low light", "grain", "motion", "angle", "frame", "lens"))
    _assert_leak_free(_build_prompt(describe_scene("resting")))
