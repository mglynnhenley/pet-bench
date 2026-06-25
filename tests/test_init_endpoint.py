"""POST /internal/init — per-sample world (re)initialization over HTTP.

The eval needs each sample to (re)initialize the world to a chosen
(scenario, seed) at runtime. ``create_app(world, world_factory=build_world)``
holds the live world on ``app.state`` and swaps it on init; presentation logs
reset. The hidden-field invariant must still hold afterward. Written first (RED).
"""

import json

from fastapi.testclient import TestClient

from petcare.config import Config
from petcare.server import build_world
from petcare.service import create_app
from petcare.world import new_world

HIDDEN_FIELDS = (
    "health", "illness_type", "illness_severity",
    "hunger", "hydration", "weight", "ate_today",
)


def _assert_no_hidden(blob):
    text = json.dumps(blob)
    for field in HIDDEN_FIELDS:
        assert field not in text, f"hidden field {field!r} leaked: {blob}"


def _app_with_factory():
    return create_app(build_world("baseline", 0), world_factory=build_world)


def test_create_app_backward_compatible_one_arg():
    # existing one-arg callers must keep working
    app = create_app(new_world(Config(), seed=0))
    client = TestClient(app)
    assert client.get("/health").status_code == 200


def test_init_without_factory_returns_not_ok():
    app = create_app(new_world(Config(), seed=0))
    client = TestClient(app)
    r = client.post("/internal/init", json={"scenario": "baseline", "seed": 1})
    assert r.status_code == 200
    assert r.json() == {"ok": False}


def test_init_rebuilds_world_for_scenario():
    client = TestClient(_app_with_factory())
    r = client.post("/internal/init", json={"scenario": "illness_episode", "seed": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["scenario"] == "illness_episode"
    assert body["seed"] == 2
    assert body["day"] == 0
    assert body["hour"] == 0
    # ground truth (internal) reflects the sick dog after init
    full = client.get("/internal/state/full").json()
    assert full["dog"]["illness_type"] is not None
    assert full["dog"]["illness_severity"] >= 1


def test_init_is_deterministic_for_identical_args():
    client = TestClient(_app_with_factory())
    client.post("/internal/init", json={"scenario": "illness_episode", "seed": 9})
    a = client.get("/internal/state/full").json()
    client.post("/internal/init", json={"scenario": "illness_episode", "seed": 9})
    b = client.get("/internal/state/full").json()
    assert a["dog"] == b["dog"]
    assert a["food_stock"] == b["food_stock"]


def test_init_different_seeds_differ_over_time():
    # advance both a while; rng-driven trajectories diverge by seed
    c1 = TestClient(_app_with_factory())
    c1.post("/internal/init", json={"scenario": "baseline", "seed": 1})
    for _ in range(5):
        c1.post("/internal/advance-day")
    s1 = c1.get("/internal/state/full").json()["tick_log"]

    c2 = TestClient(_app_with_factory())
    c2.post("/internal/init", json={"scenario": "baseline", "seed": 99999})
    for _ in range(5):
        c2.post("/internal/advance-day")
    s2 = c2.get("/internal/state/full").json()["tick_log"]

    assert s1 != s2


def test_agent_facing_endpoints_hide_hidden_after_init():
    client = TestClient(_app_with_factory())
    client.post("/internal/init", json={"scenario": "illness_episode", "seed": 4})
    for path in ("/system/status", "/account", "/supplies"):
        _assert_no_hidden(client.get(path).json())


def test_init_resets_presentation_logs():
    client = TestClient(_app_with_factory())
    # dirty the presentation state
    client.post("/care/note", json={"text": "marker"})
    client.post("/care/feed")
    client.post("/vet/consult", json={"symptoms": "x"})
    log_before = client.get("/care/log").json()
    assert log_before["notes"] and log_before["actions"]

    client.post("/internal/init", json={"scenario": "baseline", "seed": 0})
    log_after = client.get("/care/log").json()
    assert log_after["notes"] == []
    assert log_after["actions"] == []
    full = client.get("/internal/state/full").json()
    assert full["vet_records"] == []
    assert full["sent_emails"] == []


def test_init_swaps_world_used_by_actions():
    # after init the world the endpoints mutate is the NEW one
    client = TestClient(_app_with_factory())
    client.post("/internal/init", json={"scenario": "baseline", "seed": 0})
    client.post("/care/schedule-next", json={"hours": 3})
    client.post("/internal/advance-next")
    full = client.get("/internal/state/full").json()
    assert full["tick"] == 3
