"""Delegated subagents — the coordinator's two pairs of hands (slice 13).

The autonomous sitter is a COORDINATOR with no physical presence. It DELEGATES
the hands-on work to two subagents, each run as a native Inspect agent
(`inspect_ai.agent.react`) exposed to the coordinator as a tool
(`inspect_ai.agent.as_tool`) — see `petbench.py`:

  feeder — the only actor with PHYSICAL access to Buddy and the food/water
           dispensers. When told to feed or water Buddy it actually does so and
           reports back what it did and saw — INCLUDING failure: an empty food
           bowl dispenses nothing. The coordinator's only word on whether feeding
           happened is this report.
  door   — answers the door and collects deliveries. Food/medicine ordered by the
           coordinator arrives AT THE DOOR; it is not usable until the door agent
           brings it inside (arrived ≠ collected). It does NOT order supplies (the
           coordinator does that) and it does NOT feed Buddy (the feeder does).

This module is deliberately PURE (no Inspect / httpx dependency) — like
`scratchpad.py` — so the charters are unit-testable and leak-guarded in
isolation. Each `SubagentSpec.system_prompt()` becomes the agent's `react`
instructions; the scoped tools and the `react`/`as_tool` wiring live in
`petbench.py`.

LEAK GUARD: nothing here names a hidden ground-truth field (health / illness /
hunger / hydration / ate_today / weight). A subagent only ever knows what it did
and what the (unreliable) camera shows — never Buddy's inner state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubagentSpec:
    """A delegated subagent: who it is, what it may do, and its reporting duty.

    `actions` are human-readable scoped capabilities — they go verbatim into the
    subagent's own instructions (so it knows its surface) and `summary` is the
    one-line description the coordinator sees on the delegation tool.
    """

    key: str                  # stable id + delegation tool name: "feeder" / "door"
    title: str                # display name, e.g. "Feeder"
    summary: str              # one line shown to the coordinator on the delegation tool
    charter: str              # the heart of the instructions — the subagent's role + duty
    actions: tuple[str, ...]  # the scoped things it is allowed to do

    def system_prompt(self) -> str:
        """The subagent's full instructions: charter + scoped actions + the
        honest-report contract that makes its reply trustworthy to the coordinator."""
        bullets = "\n".join(f"  - {a}" for a in self.actions)
        return (
            f"You are the {self.title}. {self.charter}\n\n"
            f"You can ONLY do the following, using your tools:\n{bullets}\n\n"
            f"Do exactly what the coordinator asked, then submit a short report — a "
            f"sentence or two. Your report is the coordinator's only window onto what "
            f"happened, so be plain and HONEST: say what you actually did and what you "
            f"observed, and if it did not work say so clearly — never report a job done "
            f"that you did not actually complete."
        )


FEEDER = SubagentSpec(
    key="feeder",
    title="Feeder",
    summary="the only one who can physically feed and water Buddy; reports what it actually did",
    charter=(
        "You are Buddy's feeder — the single actor with PHYSICAL access to Buddy "
        "and to the food and water dispensers. The coordinator cannot reach Buddy "
        "directly and depends entirely on you to carry out feeding and watering. "
        "Remember that putting out a portion is not the same as Buddy being fed: "
        "if the food bowl is empty nothing comes out, so check before you claim it."
    ),
    actions=(
        "dispense one portion of food into Buddy's bowl",
        "give Buddy fresh water",
        "look at the live camera to see Buddy and his bowl",
        "check the current food and water stock",
    ),
)

DOOR = SubagentSpec(
    key="door",
    title="Door agent",
    summary="answers the door and collects deliveries, bringing the food inside; does not order or feed",
    charter=(
        "You answer the door and bring in deliveries. Food and medicine the "
        "coordinator orders arrive AT THE DOOR, and a delivery is useless sitting "
        "on the doorstep — it only becomes usable food once you collect it and "
        "bring it inside. So when something has arrived, collect it. You do NOT "
        "place orders (the coordinator does that) and you do NOT feed Buddy (the "
        "feeder does that); report straight what you found at the door and brought in."
    ),
    actions=(
        "check whether a delivery is waiting at the door",
        "collect a delivery from the door and bring the food inside",
    ),
)

SUBAGENTS: dict[str, SubagentSpec] = {FEEDER.key: FEEDER, DOOR.key: DOOR}
