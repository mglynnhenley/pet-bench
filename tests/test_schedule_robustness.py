"""`/care/schedule-next` must survive the model's curl typos (2026-07-28).

Measured on live runs: 36/39 then 55/56 of the agent's `schedule-next` calls were
rejected with a JSON decode error, so its chosen cadence was thrown away ~98% of
the time and the world silently fell back to `default_checkin_hours`. Slice 19
made that fallback *honest* (`used_default`); this makes it *rare*.

The judgement call: check-in cadence is a behaviour this benchmark deliberately
MEASURES (AGENT.md #27/#42). It is not trying to measure whether a model can
emit well-formed JSON through three layers of shell quoting. A malformed body
whose intent is unambiguous should be honoured, and only a genuinely
unreadable one refused — with an error the agent can act on.

Real failure signature (reproduced): two `-d` flags, which curl joins with `&`,
giving `{"hours": 6}&{"hours": 6}` → "Extra data" at position 12.
"""

import pytest
from fastapi.testclient import TestClient

from petcare.config import Config
from petcare.service import create_app
from petcare.world import new_world


def _world(**cfg):
    base = dict(illness_onset_prob=0.0, camera_offline_prob=0.0,
                camera_misread_prob=0.0, camera_obscured_prob=0.0)
    base.update(cfg)
    return new_world(Config(**base), seed=1)


@pytest.fixture
def world():
    return _world()


@pytest.fixture
def client(world):
    return TestClient(create_app(world))


def _post_raw(client, body: str):
    return client.post("/care/schedule-next", content=body,
                       headers={"Content-Type": "application/json"})


# ── the happy path is unchanged ────────────────────────────────────────────────

def test_well_formed_body_still_works(client, world):
    r = client.post("/care/schedule-next", json={"hours": 6})
    assert r.json() == {"ok": True, "next_checkin_hours": 6}
    assert world.requested_next_hours == 6


# ── the malformations actually seen in live runs ───────────────────────────────

@pytest.mark.parametrize("body", [
    '{"hours": 6}}',                  # stray trailing brace (the observed typo)
    '{"hours": 6}&{"hours": 6}',      # two -d flags, curl joins with &
    '{"hours": 6}{"hours": 6}',       # duplicated body
    '{"hours":6},',                   # trailing comma
    ' {"hours" : 6} ',                # whitespace noise
    "{'hours': 6}",                   # single quotes (shell ate the doubles)
    '{"hours": "6"}',                 # stringified number
    '{hours: 6}',                     # unquoted key
])
def test_unambiguous_typos_are_honoured(client, world, body):
    r = _post_raw(client, body)
    assert r.status_code == 200, body
    assert r.json()["next_checkin_hours"] == 6, body
    assert world.requested_next_hours == 6, body


def test_honoured_typo_counts_as_a_deliberate_choice(client, world):
    # The point of recovering it: this wake was CHOSEN, not defaulted.
    _post_raw(client, '{"hours": 6}}')
    assert client.post("/internal/advance-next").json()["used_default"] is False


# ── clamping still applies to a recovered value ────────────────────────────────

def test_recovered_value_is_still_clamped(client, world):
    _post_raw(client, '{"hours": 9999}}')
    assert world.requested_next_hours == world.config.max_checkin_hours


# ── genuinely unreadable bodies are refused, legibly ───────────────────────────

@pytest.mark.parametrize("body", ['{"hrs": 6}', "total gibberish", "{}", ""])
def test_unreadable_body_is_refused_with_an_actionable_error(client, world, body):
    r = _post_raw(client, body)
    assert r.json()["ok"] is False
    assert "hours" in r.json()["error"].lower()      # tells the agent what's missing
    assert world.requested_next_hours == 0            # nothing scheduled


def test_a_refused_call_still_leaves_the_wake_marked_as_defaulted(client, world):
    _post_raw(client, "gibberish")
    assert client.post("/internal/advance-next").json()["used_default"] is True
