"""petcare/mailflow.py — when unsolicited mail arrives.

The cadence layer above petcare/personas.py. Personas says WHO writes and WHAT
sort of mail exists; this module decides WHICH unsolicited triggers fire on a
given agent wake. Pure and deterministic: it takes the elapsed window, a few
world-derived signals, the cadence knobs (Config.cadence / DEFAULT_EMAIL_CADENCE),
the episode owner's chattiness, and an injected rng, and returns the list of
trigger keys to deliver this wake (capped). No I/O, no model calls, no world
import — the solver computes `signals` from world state and feeds them in.

Design intent (AGENT.md 7b): email frequency is a behavioural readout, not flat
noise. Two firing modes —
  • event-driven (the meaningful mail): `neighbour_concern` when the dog has been
    in audible distress long enough; `vet_followup` after a real consult; and the
    owner reacting — `owner_alarmed` (distressed + angry) when the dog is
    genuinely unwell AND the owner has been tipped off, else an `owner_checkin` /
    `owner_anxious_nudge` when the agent has simply gone quiet.
  • cadence-driven (ambient texture): supplier promo, helpline tip, spam, the odd
    owner postcard — low per-day probabilities, behaviour-independent.

So a caring, communicative sitter sees a quiet inbox; a neglectful or silent one
sees escalating pressure — culminating in an angry owner who heard from the
neighbour, not from them.

INVARIANT: the owner never learns hidden state here. `owner_alarmed` is *gated*
on real sickness + a plausible channel, but the email it produces only says "a
neighbour got in touch, worried" — never the dog's hidden fields (the firing is
grounded; the content is not).
"""

from __future__ import annotations

import random

from petcare.config import DEFAULT_EMAIL_CADENCE

# Every trigger key this scheduler can emit. Cross-checked against
# personas.TRIGGERS in the tests so the two modules can't drift apart.
EVENT_KEYS: tuple[str, ...] = (
    "neighbour_concern", "vet_followup",
    "owner_alarmed", "owner_anxious_nudge", "owner_checkin",
)


def _bernoulli(rng: random.Random, p: float) -> bool:
    return rng.random() < max(0.0, min(1.0, p))


def _compound(p_day: float, days: float) -> float:
    """P(at least one event over `days`) given a per-day probability."""
    p_day = max(0.0, min(1.0, p_day))
    return 1.0 - (1.0 - p_day) ** max(0.0, days)


def schedule_inbound(elapsed_hours: float, signals: dict | None = None,
                     cadence: dict | None = None, owner_chattiness: float = 1.0,
                     rng: random.Random | None = None) -> list[str]:
    """Return the unsolicited trigger keys to deliver on this wake.

    ``signals`` (all optional) is the solver's world-derived projection:
      - ``distress_hours``: hours the dog was in audible-distress band this window
      - ``neighbour_concern_cooldown``: bool, neighbour already wrote recently
      - ``consult_age_hours``: age of a consult awaiting follow-up, else None
      - ``owner_alarm_warranted``: bool, dog genuinely unwell AND owner tipped off
      - ``owner_silence_hours``: hours since the agent last emailed the owner

    Event-driven mail is decided first (and keeps priority under the per-wake
    cap); ambient mail fills the remainder. Deterministic for a given ``rng``.
    """
    cadence = cadence or DEFAULT_EMAIL_CADENCE
    signals = signals or {}
    rng = rng if rng is not None else random.Random()
    days = max(0.0, elapsed_hours) / 24.0
    alarm = bool(signals.get("owner_alarm_warranted"))
    fired: list[str] = []

    # ── event-driven ────────────────────────────────────────────────────────
    # neighbour concern: sustained audible distress, outside its cooldown
    distress_hours = signals.get("distress_hours", 0) or 0
    if (distress_hours >= cadence["neighbour_distress_min_hours"]
            and not signals.get("neighbour_concern_cooldown", False)
            and _bernoulli(rng, cadence["neighbour_concern_prob"])):
        fired.append("neighbour_concern")

    # vet follow-up: a real consult has aged past the delay
    consult_age = signals.get("consult_age_hours")
    if (consult_age is not None and consult_age >= cadence["vet_followup_delay_hours"]
            and _bernoulli(rng, cadence["vet_followup_prob"])):
        fired.append("vet_followup")

    # owner — at most ONE piece, by escalation: alarmed (sick + tipped off)
    # outranks a quiet-agent nudge.
    if alarm:
        if _bernoulli(rng, cadence["owner_alarm_prob"]):
            fired.append("owner_alarmed")
    else:
        silence = signals.get("owner_silence_hours")
        if silence is not None:
            threshold = cadence["owner_silence_hours"] / max(0.1, owner_chattiness)
            if silence >= threshold and _bernoulli(
                    rng, _compound(cadence["owner_nudge_prob_per_day"] * owner_chattiness, days)):
                fired.append("owner_anxious_nudge" if silence >= 2 * threshold
                             else "owner_checkin")

    # ── cadence-driven (ambient) ─────────────────────────────────────────────
    # Owner warmth is suppressed while an alarm is warranted (a cheery postcard
    # mid-crisis would be jarring and unrealistic).
    for key, p_day in cadence.get("ambient_per_day", {}).items():
        if alarm and key.startswith("owner_"):
            continue
        scaled = p_day * (owner_chattiness if key.startswith("owner_") else 1.0)
        if _bernoulli(rng, _compound(scaled, days)):
            fired.append(key)

    return fired[:cadence.get("max_per_wake", 3)]
