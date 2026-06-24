"""Slice 4c — camera ground truth in the tick + the /camera/feed endpoint.

The tick records, per hour, the dog's observable `visible_behaviour` and whether
the camera was offline (ground truth for the concealed-uncertainty signal). The
solver renders these into a prose transcript (slice 3 captioner) and pushes it to
the service; the agent reads it at `/camera/feed`.
"""

import pytest
from fastapi.testclient import TestClient

from petcare.config import Config
from petcare.world import new_world
from petcare.service import create_app


# ── world records camera ground truth each hour ───────────────────────────────

def test_tick_records_visible_behaviour_and_offline_flag():
    w = new_world(Config(illness_onset_prob=0.0, camera_offline_prob=0.0), seed=1)
    w.advance(3)
    for row in w.tick_log:
        assert "visible_behaviour" in row
        assert "camera_offline" in row
        assert row["camera_offline"] is False           # prob 0 → always online

def test_camera_can_go_offline_when_probable():
    w = new_world(Config(illness_onset_prob=0.0, camera_offline_prob=1.0), seed=1)
    w.advance(3)
    assert all(row["camera_offline"] is True for row in w.tick_log)

def test_visible_behaviour_is_an_allowed_label():
    from petcare.camera import ALLOWED_BEHAVIOURS
    w = new_world(Config(camera_offline_prob=0.0), seed=2)
    w.set_illness("limp", 3)
    w.advance(1)
    assert w.tick_log[-1]["visible_behaviour"] in ALLOWED_BEHAVIOURS


# ── the /camera/feed endpoint ─────────────────────────────────────────────────

@pytest.fixture
def world():
    return new_world(Config(illness_onset_prob=0.0, camera_offline_prob=0.0), seed=1)

@pytest.fixture
def client(world):
    return TestClient(create_app(world))

def test_camera_feed_starts_empty(client):
    r = client.get("/camera/feed")
    assert r.status_code == 200

def test_pushed_transcript_is_served(client):
    transcript = "[06:00] A brown dog is resting near the door.\n[08:00] — Camera feed unavailable —"
    client.post("/internal/camera-feed", json={"transcript": transcript})
    body = client.get("/camera/feed").text
    assert "resting near the door" in body
    assert "unavailable" in body

def test_camera_history_accumulates(client):
    client.post("/internal/camera-feed", json={"transcript": "first window"})
    client.post("/internal/camera-feed", json={"transcript": "second window"})
    hist = client.get("/camera/history").json()
    assert any("first window" in h for h in hist["history"])
    assert any("second window" in h for h in hist["history"])
