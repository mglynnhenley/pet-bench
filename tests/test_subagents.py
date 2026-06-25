"""Slice 13 — delegated subagents (the feeder + the door agent).

The autonomous sitter is a COORDINATOR with no hands of its own. It delegates
the physical work to two subagents, built as native Inspect agents
(`inspect_ai.agent.react`) and exposed to the coordinator as delegation tools
(`as_tool`), named `feeder` / `door`:

  feeder — the only one who can physically feed/water Buddy; reports honestly
           what it did, including failure (an empty bowl dispenses nothing).
  door   — answers the door and collects deliveries, bringing the food inside so
           it becomes usable stock (arrived ≠ collected). It does NOT order
           supplies (the coordinator does) and does NOT feed Buddy.

The pure contract (`petcare.subagents`) is tested here with no model; the
delegation-tool wiring (`petbench._delegation_tool` / the scoped tool sets) is
tested by introspecting the generated Inspect tools — neither needs an API key.
"""

import pytest

from petcare.subagents import DOOR, FEEDER, SUBAGENTS, SubagentSpec

# The dog's hidden ground-truth fields — must never reach any subagent prompt
# (same leak guard the personas/camera modules hold).
FORBIDDEN = ("health", "illness", "hunger", "hydration", "ate_today", "weight")


# ── the two specs exist and are wired ─────────────────────────────────────────

def test_two_subagents_registered():
    assert FEEDER.key == "feeder"
    assert DOOR.key == "door"
    assert SUBAGENTS == {"feeder": FEEDER, "door": DOOR}
    assert isinstance(FEEDER, SubagentSpec) and isinstance(DOOR, SubagentSpec)


def test_feeder_has_physical_food_and_water_actions():
    actions = " ".join(FEEDER.actions).lower()
    assert "food" in actions or "portion" in actions
    assert "water" in actions
    assert "physical" in FEEDER.charter.lower()


def test_door_collects_deliveries_and_does_not_order_or_feed():
    actions = " ".join(DOOR.actions).lower()
    assert "door" in actions
    assert "collect" in actions
    # the door agent neither orders supplies nor feeds Buddy
    assert "order" not in actions
    assert "feed" not in actions
    charter = DOOR.charter.lower()
    assert "do not place orders" in charter or "not" in charter and "order" in charter


# ── the instructions are a faithful, honest charter ───────────────────────────

@pytest.mark.parametrize("spec", [FEEDER, DOOR])
def test_system_prompt_names_role_and_demands_honest_report(spec):
    sp = spec.system_prompt()
    assert spec.title.lower() in sp.lower()
    for action in spec.actions:               # every scoped action is advertised
        assert action in sp
    assert "report" in sp.lower()
    assert "honest" in sp.lower()


@pytest.mark.parametrize("spec", [FEEDER, DOOR])
def test_subagent_prompt_never_leaks_hidden_truth(spec):
    blob = (spec.system_prompt() + " " + spec.summary).lower()
    for word in FORBIDDEN:
        assert word not in blob, f"{word!r} leaked into the {spec.key} charter"


# ── the delegation tools are native Inspect agents-as-tools ───────────────────

def _tooldef(tool):
    from inspect_ai.tool import ToolDef
    return ToolDef(tool)


@pytest.mark.parametrize("spec", [FEEDER, DOOR])
def test_delegation_tool_is_named_for_the_subagent_with_an_input(spec):
    import petbench

    td = _tooldef(petbench._delegation_tool(spec))
    assert td.name == spec.key                       # the coordinator calls feeder()/door()
    assert "input" in td.parameters.properties       # passes its instruction as `input`


def test_feeder_tool_set_is_feed_water_look_check():
    import petbench

    names = {_tooldef(t).name for t in petbench._feeder_tools()}
    assert names == {"feed", "water", "look", "check_stock"}


def test_door_tool_set_is_check_and_collect_only():
    import petbench

    names = {_tooldef(t).name for t in petbench._door_tools()}
    assert names == {"check_door", "collect"}
    # the door agent has no ordering tool — ordering is the coordinator's job
    assert "order_food" not in names
