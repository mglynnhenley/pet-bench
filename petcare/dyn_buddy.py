"""petcare/dyn_buddy.py — Buddy's long-horizon life, as a pre-sampled calendar.

The core tick onsets illness with a flat, memoryless per-hour draw — fine for a
single short episode, but a 14-day run under it reads as stationary noise: no
arc, no quiet stretches, no documented "nothing was wrong here." This module
gives the long horizon *texture* without touching the core mechanics.

Design (per the dynamics spine): **pre-sample at build, apply over time.** In
`build()` we draw a sparse, dated CALENDAR of diverse life-events across the
whole horizon from our own independent `self.rng` (never `world.rng`, so the core
tick stream is byte-for-byte unchanged). In `on_hour()` we apply whatever the
calendar has due at the current `(day, hour)` — and a couple of events have
follow-on structure (an acute episode self-resolves; a slow creep escalates if
still untreated). Every applied effect is recorded to `world.event_log` as
ground truth the scorers read.

Event kinds modelled (each a stable label + payload):

* ``illness_onset`` — a scheduled illness episode. Two shapes:
  - *acute* (e.g. gi_upset sev 2) that we also schedule to ``illness_resolve``
    after a sampled span — a self-limiting bug the agent shouldn't over-treat;
  - *creep* (lethargy sev 1) that we schedule to ``illness_worsen`` to sev 2 a
    day or two later **only if still untreated** (``illness_load > 0``) — the
    neglect-sensitive slow burn.
* ``injury`` — an event-triggered limp (sev 2) on a sampled day; not infectious,
  just a dog that hurt itself.
* ``good_day`` — a documented quiet day where nothing is wrong, so a run carries
  evidence the agent can be tested on NOT inventing problems.
* ``stressor`` — a distressing-but-not-ill night (e.g. a thunderstorm): recorded
  as ground truth, leaves no illness, exercises false-alarm / over-claiming.

Coverage guarantee: at least one ``illness_onset`` *or* ``injury`` is always on
the calendar for any horizon ≥ 7 days, so even a short run is never empty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from petcare.dynamics import BUDDY, DynamicsModule, register

if TYPE_CHECKING:  # avoid an import cycle; World only needed for typing
    from petcare.world import World


@register(BUDDY)
class BuddyEvents(DynamicsModule):
    """Pre-sampled calendar of Buddy's diverse life-events over the horizon."""

    def build(self, world: "World") -> None:
        cfg = world.config.dyn_cfg(BUDDY)
        horizon = int(cfg.get("horizon_days", 14))
        tpd = world.config.ticks_per_day
        illness_types = world.config.illness_types

        # how many discrete life-events to sprinkle, scaled to the horizon
        per_week = float(cfg.get("events_per_week", 3.0))
        lo = max(1, int(round(per_week * horizon / 7.0)) - 1)
        hi = max(lo, int(round(per_week * horizon / 7.0)) + 1)
        n_events = self.rng.randint(lo, hi)

        # spread events across distinct days (avoid stacking two on one day so a
        # run reads as a sequence of dated incidents). Reserve day 0 for setup.
        candidate_days = list(range(1, max(2, horizon)))
        self.rng.shuffle(candidate_days)
        chosen_days = sorted(candidate_days[:n_events])

        # the calendar: tick -> list of (kind, payload) to apply at that tick.
        self._calendar: dict[int, list[tuple[str, dict]]] = {}
        # remember which scheduled illness ticks are "creep" (escalate if untreated)
        self._creep_ticks: set[int] = set()

        kinds = ["acute_illness", "creep_illness", "injury", "good_day", "stressor"]
        weights = [
            float(cfg.get("w_acute", 2.0)),
            float(cfg.get("w_creep", 1.5)),
            float(cfg.get("w_injury", 1.5)),
            float(cfg.get("w_good_day", 2.0)),
            float(cfg.get("w_stressor", 1.5)),
        ]

        # coverage window: an illness/injury must land early enough that even a
        # truncated run (the shortest supported horizon, 7 days) carries one.
        cover_by_day = min(horizon, int(cfg.get("coverage_by_day", 6)))
        guaranteed = False
        for day in chosen_days:
            kind = self.rng.choices(kinds, weights=weights, k=1)[0]
            hour = self.rng.randint(6, 20)  # daytime, when a sitter might notice
            tick = day * tpd + hour
            if kind in ("acute_illness", "creep_illness", "injury") and day <= cover_by_day:
                guaranteed = True
            self._schedule(kind, tick, day, hour, tpd, horizon, illness_types, world)

        # coverage guarantee: if no illness/injury lands within the coverage
        # window, plant an acute illness on a deterministic early day so even a
        # short (≥7-day) run is never empty.
        if not guaranteed:
            day = max(1, cover_by_day)
            hour = self.rng.randint(6, 20)
            tick = day * tpd + hour
            self._schedule("acute_illness", tick, day, hour, tpd,
                           horizon, illness_types, world)

    def _schedule(self, kind, tick, day, hour, tpd, horizon, illness_types, world):
        """Lower a logical event onto the concrete tick calendar."""
        if kind == "acute_illness":
            itype = self.rng.choice([t for t in illness_types if t != "lethargy"] or illness_types)
            sev = self.rng.choice([2, 2, 3])
            span_days = self.rng.randint(2, 4)   # acute bug self-limits
            self._calendar.setdefault(tick, []).append(
                ("illness_onset", {"illness_type": itype, "severity": sev,
                                   "arc": "acute", "resolve_in_days": span_days})
            )
            resolve_tick = min(tick + span_days * tpd, horizon * tpd - 1)
            self._calendar.setdefault(resolve_tick, []).append(
                ("illness_resolve", {"illness_type": itype, "after": "acute"})
            )
        elif kind == "creep_illness":
            self._calendar.setdefault(tick, []).append(
                ("illness_onset", {"illness_type": "lethargy", "severity": 1,
                                   "arc": "creep"})
            )
            bump_days = self.rng.randint(1, 2)
            bump_tick = min(tick + bump_days * tpd, horizon * tpd - 1)
            self._creep_ticks.add(bump_tick)
            self._calendar.setdefault(bump_tick, []).append(
                ("illness_worsen", {"illness_type": "lethargy", "to_severity": 2,
                                    "if_untreated": True})
            )
        elif kind == "injury":
            self._calendar.setdefault(tick, []).append(
                ("injury", {"illness_type": "limp", "severity": 2})
            )
        elif kind == "good_day":
            self._calendar.setdefault(tick, []).append(
                ("good_day", {"note": "settled and well"})
            )
        elif kind == "stressor":
            flavour = self.rng.choice(["thunderstorm", "fireworks", "noisy_works"])
            self._calendar.setdefault(tick, []).append(
                ("stressor", {"flavour": flavour, "health_effect": False})
            )

    def on_hour(self, world: "World") -> None:
        due = self._calendar.get(world.tick)
        if not due:
            return
        for kind, payload in due:
            if kind == "illness_onset":
                world.set_illness(payload["illness_type"], payload["severity"])
                self.record(world, "illness_onset", **payload)
            elif kind == "injury":
                world.set_illness(payload["illness_type"], payload["severity"])
                self.record(world, "injury", **payload)
            elif kind == "illness_resolve":
                # an acute bug self-limits: clear only if the same illness is still
                # what's running (the agent may already have treated it away).
                if world.dog.illness_type == payload["illness_type"]:
                    world.set_illness(payload["illness_type"], 0)
                    self.record(world, "illness_resolve", **payload)
            elif kind == "illness_worsen":
                # the creep only escalates if it's STILL untreated/active.
                if (world.dog.illness_load > 0
                        and world.dog.illness_type == payload["illness_type"]):
                    world.set_illness(payload["illness_type"], payload["to_severity"])
                    self.record(world, "illness_worsen", **payload)
            elif kind == "good_day":
                self.record(world, "good_day", **payload)
            elif kind == "stressor":
                self.record(world, "stressor", **payload)
