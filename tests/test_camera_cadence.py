"""Camera cadence + detail (user request 2026-06-24):
  • the feed is windowed every 2 hours (not 6), so the transcript has a line
    roughly every two sim-hours;
  • the captioner prompt asks for a richer, more detailed description — while
    still never leaking hidden-state words or digits.
"""

import petbench
from episode_runner import _windows_from_rows as runner_windows
from petcare.camera import _build_prompt, describe_scene


def _rows(n, start_hour=6):
    return [{"hour": start_hour + i, "day": 0, "camera_shown": "resting",
             "camera_offline": False, "camera_obscured": False} for i in range(n)]


def test_solver_windows_are_every_two_hours_by_default():
    windows = petbench._windows_from_rows(_rows(6, start_hour=6))
    # 6 hourly rows / 2h windows -> 3 windows, stepping by 2 hours
    assert len(windows) == 3
    assert [w.timestamp for w in windows] == ["06:00", "08:00", "10:00"]


def test_runner_windows_are_every_two_hours_by_default():
    windows = runner_windows(_rows(4, start_hour=6))
    assert len(windows) == 2  # 4 hourly rows / 2h windows


def test_captioner_prompt_asks_for_detail_without_leaking():
    prompt = _build_prompt(describe_scene("resting"))
    low = prompt.lower()
    # richer than the old "a sentence or two": explicitly asks for detail
    assert "detail" in low
    # still leak-safe: no hidden-state words, no digits
    for w in ("health", "illness", "hunger", "hydration"):
        assert w not in low
    assert not any(ch.isdigit() for ch in prompt)
