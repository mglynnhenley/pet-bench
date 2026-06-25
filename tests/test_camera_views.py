"""Camera views (2026-06-25): different functions for different information.

A real camera app isn't one passive feed — it's an effort ladder:
  • /camera/live   — the current moment only (a glance);
  • /camera/feed   — the span-adaptive recap since last check (slice 12);
  • /camera/events — the structured clip timeline (timestamps + tags + density),
                     NO prose, so *absence* of activity is legible cheaply;
  • /camera/history— scrollable past recaps.

These tests pin the new live + events surfaces. Both stay observable-only — no
hidden-state field ever crosses — and events needs no LLM (pure, server-side).
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from petcare.config import Config
from petcare.world import new_world
from petcare.service import create_app
from petcare.camera import render_live_snapshot, events_timeline


class FakeModel:
    def __init__(self):
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "the dog is in the frame"


def _row(hour, behaviour="resting", offline=False, obscured=False, day=0):
    return {"hour": hour, "day": day, "camera_shown": behaviour,
            "camera_offline": offline, "camera_obscured": obscured}


# ── render_live_snapshot: the current moment (latest hour) ────────────────────

def test_live_snapshot_describes_latest_hour_only():
    fake = FakeModel()
    rows = [_row(12, "pacing"), _row(13, "drinking")]
    snap = asyncio.run(render_live_snapshot(fake, rows))
    assert snap.startswith("[13:00]")       # the latest hour, not the whole span
    assert len(fake.prompts) == 1


def test_live_snapshot_offline_emits_marker_without_model_call():
    fake = FakeModel()
    snap = asyncio.run(render_live_snapshot(fake, [_row(13, offline=True)]))
    assert "unavailable" in snap.lower()
    assert fake.prompts == []


def test_live_snapshot_prompt_is_leak_safe():
    fake = FakeModel()
    asyncio.run(render_live_snapshot(fake, [_row(13, "lethargic")]))
    low = fake.prompts[0].lower()
    for w in ("health", "illness", "hunger", "hydration"):
        assert w not in low
    assert not any(ch.isdigit() for ch in fake.prompts[0])


# ── events_timeline: structured, prose-free, density-bearing ──────────────────

def test_events_timeline_tags_clips_and_summarises():
    rows = ([_row(6, "resting")] * 3 + [_row(9, "pacing")] * 2
            + [_row(11, offline=True)])
    tl = events_timeline(rows)
    tags = [e["tag"] for e in tl["events"]]
    assert tags == ["resting", "pacing", "offline"]
    assert tl["events"][0]["duration_h"] == 3
    assert tl["summary"]["events"] == 1        # one motion clip
    assert tl["summary"]["offline_hours"] == 1


def test_events_timeline_carries_no_prose_and_no_hidden_state():
    rows = [_row(6, "lethargic"), _row(7, "limping")]
    tl = events_timeline(rows)
    blob = str(tl).lower()
    for w in ("health", "illness_severity", "hunger", "hydration", "weight"):
        assert w not in blob


# ── the endpoints ─────────────────────────────────────────────────────────────

@pytest.fixture
def world():
    return new_world(Config(illness_onset_prob=0.0, camera_offline_prob=0.0), seed=1)


@pytest.fixture
def client(world):
    return TestClient(create_app(world))


def test_live_endpoint_serves_pushed_snapshot(client):
    client.post("/internal/camera-live", json={"transcript": "[13:00] a dog at the bowl"})
    body = client.get("/camera/live").text
    assert "a dog at the bowl" in body


def test_live_endpoint_starts_with_a_fallback(client):
    r = client.get("/camera/live")
    assert r.status_code == 200
    assert r.text  # non-empty fallback, never a server error


def test_events_endpoint_reads_the_tick_timeline(client, world):
    world.set_illness("limp", 3)
    world.advance(4)
    payload = client.get("/camera/events").json()
    assert "events" in payload and "summary" in payload
    assert payload["events"]                      # at least one clip
    # observable-only: no hidden field leaks through the timeline
    blob = str(payload).lower()
    for w in ("health", "illness_severity", "hunger", "hydration", "weight"):
        assert w not in blob


def test_events_endpoint_respects_hours_window(client, world):
    world.advance(30)
    short = client.get("/camera/events", params={"hours": 6}).json()
    covered = sum(e["duration_h"] for e in short["events"])
    assert covered <= 6
