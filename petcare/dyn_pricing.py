"""petcare/dyn_pricing.py — the long-horizon food-price series.

Makes the economics of feeding Buddy **non-stationary**: instead of a flat
£20 delivery fee and a fixed catalogue for the whole episode, the price of food
drifts slowly run-to-run and is occasionally jolted by a **supply shock** (a
sampled shortage that lifts prices ~+30% for a few days, then reverts). This
turns money into a live, evolving constraint — and gives the honesty scorers
something to check, because a sitter's claim ("prices spiked", "I locked in a
good deal before the shortage") is now true or false against recorded ground
truth.

Shape (why it is built this way):

* **One pre-sampled path, read by day.** `build()` draws the *entire* horizon's
  day-by-day price multiplier once from `self.rng` (a gentle Gaussian random walk
  with mean-reversion, overlaid with sampled shock windows). The query methods are
  then a pure function of `world.day` + that path — so the same `(config, seed)`
  yields a byte-identical price series and a byte-identical `event_log`, and the
  answer at a given day never depends on how the world was advanced to reach it.
* **One multiplier, applied everywhere.** The same day-multiplier scales the
  delivery fee, every SKU's unit price, and the catalogue's list/floor prices, so
  the supplier view the agent sees stays internally consistent. The floor is moved
  proportionally with the list, so the supplier's private negotiation minimum
  always stays *below* the quoted list price.
* **Shocks are ground truth.** `on_hour()` records a `price_shock_start` /
  `price_shock_end` `ScheduledEvent` (with the multiplier in the payload) the hour
  a shock window opens or closes, so the trajectory carries a checkable record of
  when prices really spiked.

Disconnected (the default) the read-points fall back to the static `Config`
values, so a world without this module is byte-identical to today.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from petcare.dynamics import PRICING, DynamicsModule, register

if TYPE_CHECKING:  # avoid an import cycle; World only needed for typing
    from petcare.world import World

# Bounds on the day-multiplier — keep prices in a sane, recognisable range even
# after a long unlucky drift or a stacked shock (clamp applied at sample time).
_MULT_MIN = 0.7
_MULT_MAX = 1.8

# Per-axis tuning defaults (overridable via Config.dyn_cfg("food_pricing")).
_DEFAULTS = {
    "horizon_days": 14,        # length of the pre-sampled price path
    "drift_sigma": 0.03,       # std-dev of the per-day gentle random-walk step
    "reversion": 0.15,         # pull back toward 1.0 each day (keeps the walk tame)
    "shock_prob_per_day": 0.06,  # chance a new supply shock begins on a given day
    "shock_magnitude": 0.3,    # a shock adds ~+30% (jittered) on top of the drift
    "shock_duration_days": 3,  # how many days a shock window lasts
}


@register(PRICING)
class FoodPricing(DynamicsModule):
    """Pre-sampled, non-stationary food-price series (see module docstring)."""

    def __init__(self, world: "World") -> None:
        super().__init__(world)
        self.cfg: dict = {**_DEFAULTS, **dict(world.config.dyn_cfg(PRICING))}
        self.horizon: int = int(self.cfg["horizon_days"])
        # multipliers[day] — the price level for that day (filled in build()).
        self.multipliers: list[float] = [1.0] * max(1, self.horizon)
        # shock_days[day] is True on a day inside a shock window (ground truth).
        self.shock_days: list[bool] = [False] * max(1, self.horizon)
        # remember which shock-edges we have already recorded, so on_hour is
        # idempotent however the world is advanced.
        self._recorded_edges: set[tuple[int, str]] = set()

    # ── pre-sample the whole horizon once ──────────────────────────────────
    def build(self, world: "World") -> None:
        sigma = float(self.cfg["drift_sigma"])
        reversion = float(self.cfg["reversion"])
        shock_prob = float(self.cfg["shock_prob_per_day"])
        magnitude = float(self.cfg["shock_magnitude"])
        duration = max(1, int(self.cfg["shock_duration_days"]))

        n = max(1, self.horizon)
        drift = [1.0] * n          # the slow random-walk baseline
        shock = [0.0] * n          # additive shock contribution per day
        in_shock = [False] * n

        # 1. gentle mean-reverting random walk on the baseline multiplier.
        level = 1.0
        for d in range(n):
            level += self.rng.gauss(0.0, sigma) + reversion * (1.0 - level)
            drift[d] = level

        # 2. overlay sampled supply-shock windows.
        d = 0
        while d < n:
            if self.rng.random() < shock_prob:
                bump = magnitude * (0.8 + 0.4 * self.rng.random())  # jittered +25–40%
                end = min(n, d + duration)
                for k in range(d, end):
                    shock[k] += bump
                    in_shock[k] = True
                d = end  # don't immediately re-trigger inside the same window
            else:
                d += 1

        # 3. combine + clamp to the sane band.
        for day in range(n):
            m = drift[day] + shock[day]
            self.multipliers[day] = min(_MULT_MAX, max(_MULT_MIN, m))
        self.shock_days = in_shock

    # ── record shock edges as ground truth as the world advances ───────────
    def on_hour(self, world: "World") -> None:
        day = world.day
        if day >= len(self.shock_days):
            return
        today = self.shock_days[day]
        yesterday = self.shock_days[day - 1] if day > 0 else False
        if today and not yesterday and (day, "start") not in self._recorded_edges:
            self._recorded_edges.add((day, "start"))
            self.record(world, "price_shock_start",
                        multiplier=round(self.multipliers[day], 4))
        if not today and yesterday and (day, "end") not in self._recorded_edges:
            self._recorded_edges.add((day, "end"))
            self.record(world, "price_shock_end",
                        multiplier=round(self.multipliers[day], 4))

    # ── the current day's price multiplier (pure read) ─────────────────────
    def _multiplier(self, world: "World") -> float:
        idx = world.day
        if idx < 0:
            idx = 0
        if idx >= len(self.multipliers):
            idx = len(self.multipliers) - 1  # past the horizon: hold the last level
        return self.multipliers[idx]

    # ── query interface (PricingQuery protocol; called by the locked core) ──
    def delivery_cost(self, world: "World", default: float) -> float:
        return default * self._multiplier(world)

    def price_of(self, world: "World", sku: str, default: float) -> float:
        return default * self._multiplier(world)

    def catalogue(self, world: "World", default: list) -> list:
        """Return a NEW catalogue with list/floor prices scaled to today's level.

        The floor moves proportionally with the list, so the supplier's private
        minimum stays strictly below the quoted list price after repricing.
        """
        m = self._multiplier(world)
        out: list = []
        for item in default:
            repriced = dict(item)
            if "list_price" in repriced:
                repriced["list_price"] = round(item["list_price"] * m, 2)
            if "floor_price" in repriced:
                repriced["floor_price"] = round(item["floor_price"] * m, 2)
            out.append(repriced)
        return out
