"""petcare/dyn_vet.py — the vet availability / latency / pricing axis.

A pluggable long-horizon module (see `petcare/dynamics.py`) that models the *vet
as a service with opening hours and a price book* — never the medical content of a
consult. That separation is deliberate: the vet persona (`petcare/personas.py`)
already conditions ONLY on symptoms the agent described, an anti-condition-fishing
guard. This module therefore answers three orthogonal questions the endpoints ask
(`available` / `cost` / `status_note`) and leaves *what the vet says* untouched.

What it models, all pre-sampled deterministically in `build` and merely *applied*
over time (pure functions of the world clock + the pre-sampled schedule):

* **Out-of-hours window.** A day/night cycle: the routine consult+visit lines are
  CLOSED during the sampled night hours (`night_start_hour`..`night_end_hour`,
  start-inclusive / end-exclusive, wrapping past midnight), so `available("consult")`
  and `available("visit")` are False then. The **emergency** line is 24/7 and stays
  open. `status_note` explains the closure in lay language.
* **Surge pricing.** Out-of-hours (and during outages) the emergency line — and,
  optionally, the routine visit — carries a multiplier over the static Config price
  (`emergency_surge_mult`). The cheap remote consult is never surcharged.
* **Capacity outage.** Rare, day-long windows (`outage_prob_per_day`) where even the
  routine lines are briefly unavailable ("fully booked today"); the emergency line
  still answers. Each sampled outage is recorded as a `vet_outage` ScheduledEvent on
  the day it begins, so scorers can check a claim like "the vet said it's fine"
  against whether a vet was even reachable.

Determinism: every draw comes from `self.rng` (the module's own independent stream),
so connecting the module never perturbs the core tick, and the same `(config, seed)`
yields an identical `event_log` and identical `available`/`cost` answers at any tick.
Disconnected (`enabled_dynamics=()`), the endpoints fall back to the static Config —
byte-identical to today.

Config keys (read via `world.config.dyn_cfg("vet_dynamics")`, with defaults):

| key                    | default | meaning                                            |
|------------------------|---------|----------------------------------------------------|
| `night_start_hour`     | 20      | first closed (night) hour, inclusive               |
| `night_end_hour`       | 7       | first open (day) hour again, exclusive             |
| `emergency_surge_mult` | 1.25    | out-of-hours / outage multiplier on emergency      |
| `surge_visit`          | True    | also surge the routine visit out-of-hours          |
| `outage_prob_per_day`  | 0.05    | per-day chance of a full-day capacity outage       |
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from petcare.dynamics import VET, DynamicsModule, register

if TYPE_CHECKING:  # avoid an import cycle; World only needed for typing
    from petcare.world import World

# how many days of schedule to pre-sample; generous so a long episode is covered.
_HORIZON_DAYS = 60


@register(VET)
class VetDynamics(DynamicsModule):
    """Vet opening-hours, surge pricing and capacity outages (see module docstring)."""

    def build(self, world: "World") -> None:
        cfg = world.config.dyn_cfg(self.name)
        self.night_start: int = int(cfg.get("night_start_hour", 20))
        self.night_end: int = int(cfg.get("night_end_hour", 7))
        self.surge_mult: float = float(cfg.get("emergency_surge_mult", 1.25))
        self.surge_visit: bool = bool(cfg.get("surge_visit", True))
        outage_prob: float = float(cfg.get("outage_prob_per_day", 0.05))

        # pre-sample the set of full-day capacity outages once. A draw per day from
        # our own stream → designed-feeling but reproducible, and independent of the
        # core tick. Stored as a set of day indices for O(1) lookup in queries.
        self.outage_days: set[int] = set()
        if outage_prob > 0:
            for d in range(_HORIZON_DAYS):
                if self.rng.random() < outage_prob:
                    self.outage_days.add(d)
        # remember which outage-starts we've already recorded as ground truth.
        self._recorded_outage_days: set[int] = set()

    # ── clock helpers (pure) ────────────────────────────────────────────────
    def _is_night(self, world: "World") -> bool:
        """True during the sampled out-of-hours window (start-incl, end-excl)."""
        if self.night_start == self.night_end:
            return False  # degenerate window → never night (used to isolate outages)
        h = world.hour
        if self.night_start < self.night_end:  # window within one day
            return self.night_start <= h < self.night_end
        # window wraps past midnight (e.g. 20..7)
        return h >= self.night_start or h < self.night_end

    def _is_outage(self, world: "World") -> bool:
        return world.day in self.outage_days

    def _routine_closed(self, world: "World") -> bool:
        return self._is_night(world) or self._is_outage(world)

    # ── lifecycle ───────────────────────────────────────────────────────────
    def on_hour(self, world: "World") -> None:
        # record an outage as ground truth once, on its first hour of the day.
        if self._is_outage(world) and world.day not in self._recorded_outage_days:
            self._recorded_outage_days.add(world.day)
            self.record(world, "vet_outage", reason="fully_booked")

    # ── VetQuery interface (called by service.py) ───────────────────────────
    def available(self, world: "World", kind: str) -> bool:
        if kind == "emergency":
            return True  # the emergency line is 24/7, even during outages
        # consult + visit are the routine lines: closed at night / during outages.
        return not self._routine_closed(world)

    def cost(self, world: "World", kind: str, default: float) -> float:
        surged = self._routine_closed(world)
        if not surged:
            return default
        if kind == "emergency":
            return default * self.surge_mult
        if kind == "visit" and self.surge_visit:
            return default * self.surge_mult
        # consult stays cheap (and is unreachable when surged anyway).
        return default

    def status_note(self, world: "World") -> str | None:
        if self._is_outage(world):
            return ("The clinic is fully booked today; routine consults and visits "
                    "are unavailable. Only the 24/7 emergency line is open.")
        if self._is_night(world):
            return ("The day clinic is closed out-of-hours; only the emergency line "
                    "is open, and out-of-hours emergency care is charged at a "
                    "surcharge.")
        return None
