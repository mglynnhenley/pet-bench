"""Tests for the uvicorn entry point ``petcare/server.py`` (slice: server).

The Inspect solver launches the pet-care system as ``uvicorn petcare.server:app``,
configured via environment variables so different scenarios/seeds can be run.
These tests are written first (RED) and drive ``server.py``.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from petcare.config import Config
from petcare.world import World


def test_build_config_baseline_returns_config():
    from petcare.server import build_config

    cfg = build_config("baseline")
    assert isinstance(cfg, Config)


def test_build_config_unknown_falls_back_to_baseline():
    from petcare.server import build_config

    # An unknown scenario name must not raise; it falls back to baseline.
    cfg = build_config("does-not-exist")
    assert isinstance(cfg, Config)
    assert cfg == build_config("baseline")


def test_make_world_from_env_respects_seed_and_scenario(monkeypatch):
    from petcare.server import build_config, make_world_from_env

    monkeypatch.setenv("PETBENCH_SEED", "7")
    monkeypatch.setenv("PETBENCH_SCENARIO", "baseline")

    world = make_world_from_env()
    assert isinstance(world, World)
    assert world.day == 0
    # config matches the named scenario
    assert world.config == build_config("baseline")


def test_make_world_from_env_defaults(monkeypatch):
    from petcare.server import build_config, make_world_from_env

    monkeypatch.delenv("PETBENCH_SEED", raising=False)
    monkeypatch.delenv("PETBENCH_SCENARIO", raising=False)

    world = make_world_from_env()
    assert isinstance(world, World)
    assert world.day == 0
    assert world.config == build_config("baseline")


def test_make_world_from_env_unknown_scenario_falls_back(monkeypatch):
    from petcare.server import build_config, make_world_from_env

    monkeypatch.setenv("PETBENCH_SCENARIO", "totally-unknown")
    world = make_world_from_env()
    assert world.config == build_config("baseline")


def test_illness_episode_starts_dog_already_ill(monkeypatch):
    # The illness_episode scenario must present bad news from hour 0 so the
    # honesty scorers get exercised even on short, budget-limited runs — it
    # can't rely on rng onset firing during the episode.
    from petcare.server import make_world_from_env

    monkeypatch.setenv("PETBENCH_SEED", "7")
    monkeypatch.setenv("PETBENCH_SCENARIO", "illness_episode")
    world = make_world_from_env()

    assert world.day == 0
    assert world.dog.illness_severity >= 1
    assert world.dog.illness_type is not None
    assert world.dog.illness_load > 0


def test_baseline_starts_dog_healthy(monkeypatch):
    from petcare.server import make_world_from_env

    monkeypatch.setenv("PETBENCH_SEED", "7")
    monkeypatch.setenv("PETBENCH_SCENARIO", "baseline")
    world = make_world_from_env()

    assert world.dog.illness_severity == 0
    assert world.dog.illness_type is None


def test_module_level_app_is_fastapi():
    from petcare.server import app

    assert isinstance(app, FastAPI)


def test_app_health_and_status(monkeypatch):
    from petcare.server import create_app, make_world_from_env

    monkeypatch.setenv("PETBENCH_SEED", "0")
    monkeypatch.setenv("PETBENCH_SCENARIO", "baseline")

    client = TestClient(create_app(make_world_from_env()))

    r = client.get("/health")
    assert r.status_code == 200

    r = client.get("/system/status")
    assert r.status_code == 200
    assert r.json()["day"] == 0
