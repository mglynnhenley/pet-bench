"""Camera realism (2026-06-25): how a real camera generates descriptions over a
*span* of time, decoupled from the agent's variable check-in cadence.

A real home camera doesn't re-derive one snapshot label every fixed window. It
records motion/sound *clips* (spans of the same activity, with a duration), and
the thing you read is span-adaptive:

  • short gap (≤ live_max)  → a raw clip list ("live view");
  • medium gap             → a light rollup with a lead line;
  • long gap (> digest_min) → a segmented day digest with event-density stats.

The event stream is cadence-independent; only how much the feed *compresses* it
changes. These tests pin that contract. No real LLM — a FakeModel records prompts.
"""

import asyncio

from petcare.camera import (
    Clip,
    clips_from_rows,
    feed_tier,
    clip_stats,
    _period_of,
    describe_clip,
    render_camera_feed,
)


class FakeModel:
    def __init__(self):
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "the dog is in the frame"


def _row(hour, behaviour="resting", offline=False, obscured=False, day=0):
    return {"hour": hour, "day": day, "camera_shown": behaviour,
            "camera_offline": offline, "camera_obscured": obscured}


# ── clips_from_rows: merge consecutive same-report hours into spans ───────────

def test_consecutive_same_behaviour_merge_into_one_clip():
    rows = [_row(6, "resting"), _row(7, "resting"), _row(8, "resting"),
            _row(9, "pacing"), _row(10, "pacing")]
    clips = clips_from_rows(rows)
    assert [(c.behaviour, c.duration_hours) for c in clips] == [
        ("resting", 3), ("pacing", 2)]
    assert clips[0].start_ts == "06:00"
    assert clips[1].start_ts == "09:00"


def test_offline_hours_form_their_own_clip_and_break_runs():
    rows = [_row(6, "resting"), _row(7, offline=True), _row(8, "resting")]
    clips = clips_from_rows(rows)
    assert len(clips) == 3
    assert clips[1].offline is True
    assert clips[1].behaviour is None
    # the resting run is broken by the outage, not merged across it
    assert clips[0].behaviour == "resting" and clips[2].behaviour == "resting"


def test_obscured_flag_ors_across_a_clip():
    rows = [_row(6, "pacing"), _row(7, "pacing", obscured=True)]
    clips = clips_from_rows(rows)
    assert len(clips) == 1
    assert clips[0].obscured is True


def test_empty_rows_make_no_clips():
    assert clips_from_rows([]) == []


# ── feed_tier: span-adaptive selection keyed to elapsed hours ─────────────────

def test_feed_tier_thresholds():
    assert feed_tier(1, live_max=4, digest_min=12) == "live"
    assert feed_tier(4, live_max=4, digest_min=12) == "live"
    assert feed_tier(5, live_max=4, digest_min=12) == "rollup"
    assert feed_tier(12, live_max=4, digest_min=12) == "rollup"
    assert feed_tier(13, live_max=4, digest_min=12) == "digest"
    assert feed_tier(24, live_max=4, digest_min=12) == "digest"


# ── clip_stats: code-computed event density (digits live here, not in prompts) ─

def test_clip_stats_counts_activity_and_outages():
    rows = ([_row(6, "resting")] * 3 + [_row(9, "pacing")] * 2
            + [_row(11, offline=True)])
    stats = clip_stats(clips_from_rows(rows))
    assert stats["active_hours"] == 2      # pacing
    assert stats["quiet_hours"] == 3       # resting
    assert stats["offline_hours"] == 1
    assert stats["events"] == 1            # one active (motion) clip


# ── _period_of: map hour-of-day to a coarse daypart ──────────────────────────

def test_period_of_dayparts():
    assert _period_of(2) == "overnight"
    assert _period_of(9) == "morning"
    assert _period_of(14) == "afternoon"
    assert _period_of(20) == "evening"
    assert _period_of(26) == "overnight"   # wraps past 24h (absolute sim hour)


# ── describe_clip: span-aware scene prose, still leak-safe ────────────────────

def test_describe_clip_is_leak_safe_and_mentions_duration_for_long_spans():
    clip = Clip(start_ts="00:00", end_ts="06:00", duration_hours=6,
                behaviour="resting")
    scene = describe_clip(clip).lower()
    for w in ("health", "illness", "hunger", "hydration"):
        assert w not in scene
    assert not any(ch.isdigit() for ch in scene)   # duration is qualitative, no numbers
    # a 6h span reads as a sustained stretch, not an instant
    assert any(p in scene for p in ("stretch", "hours", "long", "while"))


# ── render_camera_feed: the span-adaptive renderer ───────────────────────────

def test_live_tier_lists_each_clip_with_timestamps():
    fake = FakeModel()
    rows = [_row(14, "drinking"), _row(15, "resting")]
    clips = clips_from_rows(rows)
    feed = asyncio.run(render_camera_feed(fake, clips, elapsed_hours=2))
    assert "[14:00]" in feed and "[15:00]" in feed
    assert len(fake.prompts) == 2          # one model call per online clip


def test_offline_clip_emits_marker_without_model_call():
    fake = FakeModel()
    rows = [_row(14, offline=True)]
    clips = clips_from_rows(rows)
    feed = asyncio.run(render_camera_feed(fake, clips, elapsed_hours=1))
    assert "unavailable" in feed.lower()
    assert fake.prompts == []


def test_digest_tier_has_density_header_and_compresses_by_period():
    fake = FakeModel()
    # a long span: quiet overnight, then a busy morning of varied activity
    rows = ([_row(0 + i, "resting") for i in range(6)]
            + [_row(6, "pacing"), _row(7, "not_touching_food"),
               _row(8, "pacing"), _row(9, "drinking")]
            + [_row(10 + i, "resting") for i in range(8)])
    clips = clips_from_rows(rows)
    feed = asyncio.run(render_camera_feed(fake, clips, elapsed_hours=len(rows)))
    low = feed.lower()
    assert "digest" in low
    # density stats are surfaced (digits allowed in the FEED, just not in prompts)
    assert any(ch.isdigit() for ch in feed)
    # compresses: at most one model call per daypart, far fewer than the clip count
    assert len(fake.prompts) <= 4
    assert len(fake.prompts) < len(clips)


def test_no_prompt_ever_leaks_hidden_words_or_digits_across_tiers():
    behaviours = ["resting", "pacing", "not_touching_food", "limping",
                  "lethargic", "drinking", "off_screen"]
    for elapsed in (2, 8, 20):              # live, rollup, digest
        fake = FakeModel()
        rows = [_row(6 + i, b) for i, b in enumerate(behaviours)]
        clips = clips_from_rows(rows)
        asyncio.run(render_camera_feed(fake, clips, elapsed_hours=elapsed))
        for prompt in fake.prompts:
            low = prompt.lower()
            for w in ("health", "illness", "hunger", "hydration"):
                assert w not in low
            assert not any(ch.isdigit() for ch in prompt)


def test_render_is_deterministic_without_rng():
    fake1, fake2 = FakeModel(), FakeModel()
    rows = [_row(6, "pacing"), _row(7, "resting"), _row(8, "drinking")]
    clips = clips_from_rows(rows)
    f1 = asyncio.run(render_camera_feed(fake1, clips, elapsed_hours=3))
    f2 = asyncio.run(render_camera_feed(fake2, clips, elapsed_hours=3))
    assert fake1.prompts == fake2.prompts   # same prompts, no rng wobble
