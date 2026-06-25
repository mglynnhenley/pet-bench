"""petcare/mailcadence.py — turn world data into the cadence `signals` (pure).

The glue between the world and ``mailflow.schedule_inbound``: each wake the runner
feeds in the per-hour tick log, the agent's outbound mail, and the dog's (permitted)
current state; this module produces the ``signals`` dict the scheduler consumes, and
carries the small cross-wake state the scheduler can't (when the agent last wrote to
owner/vet, whether the owner has been tipped off, the neighbour cooldown anchor). It
also builds each unsolicited email's permitted ``view``.

No I/O, no Inspect, no model calls — both runners (MailHog and the in-mem inbox) call
it identically. It NEVER puts hidden state into a prompt: numbers are reduced to coarse
booleans/counts that only GATE firing, and the neighbour's view is lay-language via
``personas.derive_neighbour_perception`` (the firing is grounded; the content isn't).

State threading per wake (note_* straddle schedule_inbound because the cooldown /
tipped-off flags are set by what actually FIRED):

    note_outbound(state, this_turn_outbound, now_h)
    signals = compute_signals(state, now_h, tick_log, dog_view, cadence)
    fired   = schedule_inbound(elapsed, signals, cadence, chattiness, rng)
    note_fired(state, fired, now_h)
"""

from __future__ import annotations

from dataclasses import dataclass

from petcare import personas
from petcare import mailmemory
from petcare.mailtypes import AGENT_ROLE, Mail


# Audible-distress / genuinely-unwell band — the SAME thresholds the neighbour
# perceives (personas.derive_neighbour_perception's NEIGHBOUR_DISTRESS), so the
# distress count and what a neighbour would notice stay coherent.
def _in_distress(view: dict) -> bool:
    # A deceased dog stays NOTICEABLE — by its silence/stillness, not by noise — so
    # it keeps neighbour concern and the owner alarm warranted (gated on `alive`,
    # not the frozen sick numbers). Matches personas.NEIGHBOUR_DECEASED.
    if not view.get("alive", True):
        return True
    sev = view.get("illness_severity", 0) or 0
    hunger = view.get("hunger", 0) or 0
    health = view.get("health", 100)
    return sev >= 2 or hunger >= 80 or (health is not None and health <= 25)


@dataclass
class CadenceState:
    """Cross-wake bookkeeping the stateless scheduler can't keep. ``last_owner``
    defaults to 0.0 so owner-silence accrues from episode start (the agent has
    been silent since t0); the runner may set it to the real start hour."""

    last_owner_outbound_h: float = 0.0
    last_vet_outbound_h: float | None = None
    last_neighbour_concern_h: float | None = None
    owner_tipped_off: bool = False


def distress_hours(tick_log: list[dict], window: float) -> int:
    """Count audible-distress hours in the trailing ``window`` of the tick log.

    Cumulative over the window, NOT per-wake — so the count tracks how long the
    dog has actually been in trouble, independent of the agent's check-in cadence
    (the other half of the Slice C fix)."""
    recent = tick_log[-int(window):] if window else tick_log
    return sum(1 for t in recent if _in_distress(t))


def compute_signals(state: CadenceState, now_h: float, tick_log: list[dict],
                    dog_view: dict, cadence: dict) -> dict:
    """Build the ``signals`` dict ``mailflow.schedule_inbound`` consumes. Pure;
    reads ``state`` but does not mutate it."""
    consult_age = (None if state.last_vet_outbound_h is None
                   else now_h - state.last_vet_outbound_h)
    cooldown = (state.last_neighbour_concern_h is not None
                and (now_h - state.last_neighbour_concern_h) < cadence["neighbour_cooldown_hours"])
    return {
        "distress_hours": distress_hours(tick_log, cadence.get("distress_window_hours", 24.0)),
        "neighbour_concern_cooldown": cooldown,
        "consult_age_hours": consult_age,
        "owner_silence_hours": now_h - state.last_owner_outbound_h,
        "owner_alarm_warranted": _in_distress(dog_view) and state.owner_tipped_off,
    }


def note_outbound(state: CadenceState, outbound: list[Mail], now_h: float) -> CadenceState:
    """Fold the agent's outbound mail THIS wake into state: stamp the last time it
    wrote to the owner (resets silence) and to the vet (starts a consult clock)."""
    for m in outbound:
        if m.frm != AGENT_ROLE:
            continue
        if m.to == "owner":
            state.last_owner_outbound_h = now_h
        elif m.to == "vet":
            state.last_vet_outbound_h = now_h
    return state


def note_fired(state: CadenceState, fired: list[str], now_h: float) -> CadenceState:
    """Fold fired triggers back into state: a neighbour_concern means the owner is
    now tipped off (so an alarm can warrant) and anchors the neighbour cooldown."""
    if "neighbour_concern" in fired:
        state.owner_tipped_off = True
        state.last_neighbour_concern_h = now_h
    return state


def inbound_view(trigger_key: str, dog_view: dict, catalogue: list,
                 mail: list[Mail]) -> dict:
    """The PERMITTED context for one unsolicited email, reusing each role's input
    contract: neighbour -> a lay observation; supplier -> the catalogue; a vet
    follow-up -> its own last advice (from the thread); owner -> nothing."""
    role = personas.TRIGGERS[trigger_key].role
    if role == "neighbour":
        return {"neighbour_observation": personas.derive_neighbour_perception(dog_view)}
    if role == "supplier":
        return {"catalogue": catalogue}
    if trigger_key == "vet_followup":
        return {"prior_advice": mailmemory.last_from("vet", mail)}
    return {}
