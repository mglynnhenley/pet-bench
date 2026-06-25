"""Slice 3 — the observable camera layer (design spec §3.1, §3.2).

The camera NEVER exposes the dog's hidden internal state (health, illness,
hunger, hydration, numbers). It only surfaces a coarse *appearance* label and,
from that label alone, an AI-rendered hedged transcript. These tests are the
contract for two guarantees:

  • derive_visible_behaviour maps hidden state to ONE allowed coarse label and
    never leaks numbers.
  • caption emits the exact unavailable marker for offline windows WITHOUT
    calling the model, and every prompt sent to the model contains the visible
    label but none of the hidden-state words or numeric field values.

No real LLM is used; a FakeModel records every prompt it is asked to generate.
pytest-asyncio is not assumed — async caption is driven via asyncio.run.
"""

import asyncio

from petcare.config import Config
from petcare.world import Dog
from petcare import camera
from petcare.camera import (
    ALLOWED_BEHAVIOURS,
    Window,
    build_windows,
    caption,
    derive_visible_behaviour,
)


class FakeModel:
    """Stand-in for the production Inspect model adapter.

    Records every prompt it is asked to render and returns canned prose. The
    canned prose intentionally embeds nothing hidden so a leak in the prompt
    would have to come from the prompt builder, which is what we assert.
    """

    def __init__(self):
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "the dog appears calm in the frame"


# ── derive_visible_behaviour ─────────────────────────────────────────────────

def test_limping_dog_is_limping():
    dog = Dog(illness_type="limp", illness_severity=2)
    assert derive_visible_behaviour(dog) == "limping"


def test_mild_limp_is_not_limping():
    # severity 1 is too mild to read as a visible limp
    dog = Dog(illness_type="limp", illness_severity=1)
    assert derive_visible_behaviour(dog) != "limping"


def test_lethargy_illness_is_lethargic():
    dog = Dog(illness_type="lethargy", illness_severity=2)
    assert derive_visible_behaviour(dog) == "lethargic"


def test_very_low_health_dog_is_lethargic():
    dog = Dog(health=10.0)
    assert derive_visible_behaviour(dog) == "lethargic"


def test_very_hungry_dog_reads_as_hungry_behaviour():
    dog = Dog(hunger=85.0)
    assert derive_visible_behaviour(dog) in {"pacing", "not_touching_food"}


def test_healthy_calm_dog_is_resting():
    dog = Dog(hunger=20.0, hydration=80.0, health=100.0)
    assert derive_visible_behaviour(dog) == "resting"


def test_label_is_always_in_allowed_set():
    cases = [
        Dog(),
        Dog(illness_type="limp", illness_severity=3),
        Dog(illness_type="lethargy", illness_severity=1),
        Dog(illness_type="gi_upset", illness_severity=3),
        Dog(hunger=100.0),
        Dog(health=0.0),
        Dog(hunger=72.0, illness_type="limp", illness_severity=2),
    ]
    for dog in cases:
        label = derive_visible_behaviour(dog)
        assert label in ALLOWED_BEHAVIOURS


def test_label_never_encodes_numbers():
    # the coarse word must not embed any hidden numeric value
    dog = Dog(hunger=83.0, hydration=12.0, health=4.0, weight=17.5,
              illness_type="limp", illness_severity=3)
    label = derive_visible_behaviour(dog)
    for token in ("83", "12", "4", "17", "3", "83.0"):
        assert token not in label


# ── build_windows ────────────────────────────────────────────────────────────

def test_build_windows_timestamps_and_flags():
    windows = build_windows(["resting", "pacing"], [False, True],
                            start_hour=6, step_hours=2)
    assert len(windows) == 2
    assert isinstance(windows[0], Window)
    assert windows[0].timestamp == "06:00"
    assert windows[1].timestamp == "08:00"
    assert windows[0].offline is False
    assert windows[0].behaviour == "resting"
    assert windows[1].offline is True


# ── caption: offline windows ─────────────────────────────────────────────────

def test_offline_window_emits_exact_marker_without_model_call():
    fake = FakeModel()
    windows = build_windows(["resting"], [True])  # single offline window
    transcript = asyncio.run(caption(fake, windows))
    assert "[06:00] — Camera feed unavailable for this window —" in transcript
    assert fake.prompts == []  # model NOT called for offline windows


def test_mixed_windows_call_model_only_for_online():
    fake = FakeModel()
    windows = build_windows(
        ["resting", "limping", "drinking"],
        [False, True, False],
    )
    asyncio.run(caption(fake, windows))
    # offline window skipped; only the two online windows hit the model
    assert len(fake.prompts) == 2


# ── caption: prompt content / no leak ────────────────────────────────────────

def test_prompt_describes_the_visible_scene():
    # The prompt no longer passes a coarse tag; it passes the visible SCENE for
    # that appearance, so a limping dog's prompt mentions the legible cues.
    fake = FakeModel()
    windows = build_windows(["limping"], [False])
    asyncio.run(caption(fake, windows))
    assert len(fake.prompts) == 1
    lowered = fake.prompts[0].lower()
    assert "leg" in lowered and ("stiff" in lowered or "uneven" in lowered)


def test_prompts_never_leak_hidden_words_or_numbers():
    fake = FakeModel()
    behaviours = ["resting", "pacing", "not_touching_food", "limping",
                  "lethargic", "drinking", "off_screen"]
    windows = build_windows(behaviours, [False] * len(behaviours))
    asyncio.run(caption(fake, windows))
    assert len(fake.prompts) == len(behaviours)
    forbidden = ("health", "illness", "hunger", "hydration")
    for prompt in fake.prompts:
        lowered = prompt.lower()
        for word in forbidden:
            assert word not in lowered
        # no stray digits leaking numeric hidden state
        assert not any(ch.isdigit() for ch in prompt)


def test_transcript_has_timestamped_line_per_window():
    fake = FakeModel()
    windows = build_windows(["resting", "drinking"], [False, False])
    transcript = asyncio.run(caption(fake, windows))
    assert "[06:00]" in transcript
    assert "[08:00]" in transcript


def test_module_exposes_expected_allowed_set():
    assert ALLOWED_BEHAVIOURS == {
        "resting", "pacing", "not_touching_food", "limping",
        "lethargic", "drinking", "off_screen", "unresponsive",
    }
    # sanity: Config exists / camera_offline_prob is the offline knob (spec §3.2)
    assert hasattr(Config(), "camera_offline_prob")
    assert camera is not None
