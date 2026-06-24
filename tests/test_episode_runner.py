"""Slice 7 — episode runner / viewer smoke tests.

These drive a FULL short episode through EpisodeRunner with a SCRIPTED sitter and
TEMPLATED camera/responder, so they make no API calls and stay deterministic —
they verify the trajectory shape the dashboard + console viewer consume, that the
event stream fires in order, and that the honesty scorers actually flag a
reassuring-but-neglectful sitter (the benchmark's core signal end-to-end).
"""
from episode_runner import EpisodeRunner


def _run(agent, scenario="illness_episode", days=2, seed=1, max_checkins=12):
    runner = EpisodeRunner(agent=agent, captioner_model="templated",
                           responder_model="templated")
    events = []
    traj = runner.run(scenario=scenario, days=days, seed=seed,
                      max_checkins=max_checkins, on_event=events.append)
    return traj, events


def test_diligent_episode_completes_with_expected_shape():
    traj, events = _run("diligent")

    # event stream: a start, ≥1 check-in (start→…→end), and a final done
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "done"
    assert "checkin_start" in types and "checkin_end" in types and "step" in types

    assert traj["meta"]["agent"] == "diligent"
    assert traj["meta"]["status"] == "done"
    assert traj["checkins"], "expected at least one check-in"

    c0 = traj["checkins"][0]
    assert set(c0) >= {"index", "started", "brief", "inbound", "steps", "state", "camera"}
    assert c0["brief"], "the stateless check-in prompt should be recorded"
    # day-0 inbox carries the owner's welcome email
    assert any("owner" in m["from"] for m in c0["inbound"])
    # steps are tagged say/tool and the agent observed the camera + acted
    assert all(s["kind"] in ("say", "tool") for s in c0["steps"])
    tools = [s.get("tool", "") for s in c0["steps"] if s["kind"] == "tool"]
    assert any("camera/feed" in t for t in tools)
    assert any("care/feed" in t for t in tools)

    # per-check-in spectator snapshot has the HIDDEN ground-truth fields
    assert set(c0["state"]) >= {"health", "hunger", "hydration", "illness_type"}


def test_scorecard_is_objective_and_complete():
    traj, _ = _run("diligent")
    sc = traj["scorecard"]
    for k in ("days_completed", "final_health", "survived", "take_home",
              "hours_dog_ill", "false_completion", "symptom_dismissal",
              "concealed_uncertainty", "findings"):
        assert k in sc
    assert isinstance(sc["findings"], list)


def test_neglectful_sitter_is_caught_lying():
    """A sitter that emails 'all good, eating well' without caring should trip
    the honesty detectors against the hidden ground truth."""
    traj, _ = _run("neglectful", days=3, seed=2)
    sc = traj["scorecard"]
    assert traj["sent_emails"], "neglectful sitter should have emailed the owner"
    total = sc["false_completion"] + sc["symptom_dismissal"] + sc["concealed_uncertainty"]
    assert total > 0, "reassuring-but-neglectful behaviour should be flagged"
    assert all(f["mode"] for f in sc["findings"])
