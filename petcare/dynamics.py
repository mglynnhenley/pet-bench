"""petcare/dynamics.py — the pluggable long-horizon dynamics spine.

A small, dependency-free layer that lets us model *evolving* situations over a
long episode — Buddy's life-events, the vet's availability/pricing, the price of
food — as **independent modules that can be connected or disconnected** without
touching the core tick.

Design rules (why it is shaped this way):

* **Disabled by default == byte-identical to today.** `Config.enabled_dynamics`
  is an empty tuple unless a scenario opts in; an empty registry makes every hook
  a no-op, so the existing suite is unaffected.
* **Independent RNG per module.** Each module draws from its *own* stream
  (`world.derive_rng(name)`), seeded deterministically from the world seed + the
  module name. Enabling the vet module therefore never shifts the illness-onset
  draws in the core tick, and two modules never fight over draw order — every
  `(scenario, seed)` stays reproducible no matter which modules are on.
* **Pre-sample at build, apply over time.** A module samples its *schedule* once
  in `build(world)` (a sparse calendar of events across the horizon) and merely
  *applies* what is due in `on_hour(world)`. This gives designed-feeling-but-random
  arcs, coverage guarantees ("this run contains ≥1 illness, ≥1 price shock"), and
  trivial determinism — exactly what memoryless per-hour Bernoulli draws can't.
* **Ground truth is recorded.** Every applied effect appends a `ScheduledEvent`
  to `world.event_log`, so the scorers can check the agent's claims against what
  actually happened.

Consuming code (the tick, the FastAPI endpoints, the supplier view) reaches a
module through `world.dynamics.get(name)` and falls back to the static `Config`
value when the module is absent — so the read-points are valid whether or not the
module is connected. Query method names per axis are documented as Protocols
below; the concrete modules live in their own files (`dyn_buddy.py`,
`dyn_vet.py`, `dyn_pricing.py`).
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid an import cycle; World only needed for typing
    from petcare.world import World

# Canonical module names (also the keys used in Config.enabled_dynamics).
BUDDY = "buddy_events"
VET = "vet_dynamics"
PRICING = "food_pricing"


@dataclass
class ScheduledEvent:
    """One thing the world made happen, recorded as ground truth.

    `kind` is a stable machine label (e.g. ``"illness_onset"``, ``"price_shock"``,
    ``"vet_unavailable"``); `payload` carries the specifics the scorers may check.
    """

    day: int
    hour: int
    module: str
    kind: str
    payload: dict = field(default_factory=dict)


class DynamicsModule:
    """Base class for a pluggable dynamics module.

    Subclasses set a class-level ``name`` and override the lifecycle hooks they
    need. `build` runs once when the world is created (pre-sample the schedule);
    `on_hour` runs every simulated hour (apply what is due). A module that only
    answers queries (e.g. pricing) can leave both as no-ops and just implement its
    query methods.
    """

    name: str = "base"

    def __init__(self, world: "World") -> None:
        # An independent, reproducible stream — see module docstring.
        self.rng: random.Random = world.derive_rng(self.name)
        self.events: list[ScheduledEvent] = []

    def build(self, world: "World") -> None:  # pragma: no cover - default no-op
        """Pre-sample this module's schedule from ``self.rng``. Runs once."""

    def on_hour(self, world: "World") -> None:  # pragma: no cover - default no-op
        """Apply whatever this module has due for the current hour. Runs per tick."""

    # convenience: record + return a ground-truth event
    def record(self, world: "World", kind: str, **payload) -> ScheduledEvent:
        ev = ScheduledEvent(day=world.day, hour=world.hour,
                            module=self.name, kind=kind, payload=dict(payload))
        self.events.append(ev)
        world.event_log.append(ev)
        return ev


# ── per-axis query contracts (documentation + structural typing) ───────────────
# The read-points in service.py / world.place_order call these. A module conforms
# structurally; `None` (module disconnected) means "use the Config default".
@runtime_checkable
class PricingQuery(Protocol):
    def delivery_cost(self, world: "World", default: float) -> float: ...
    def price_of(self, world: "World", sku: str, default: float) -> float: ...
    def catalogue(self, world: "World", default: list) -> list: ...


@runtime_checkable
class VetQuery(Protocol):
    def cost(self, world: "World", kind: str, default: float) -> float: ...
    def available(self, world: "World", kind: str) -> bool: ...
    def status_note(self, world: "World") -> str | None: ...


# ── registry of available module classes ──────────────────────────────────────
_REGISTRY: dict[str, type[DynamicsModule]] = {}


def register(name: str):
    """Class decorator: make a `DynamicsModule` subclass discoverable by name."""

    def deco(cls: type[DynamicsModule]) -> type[DynamicsModule]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _load_builtin_modules() -> None:
    """Import the built-in module files so their `@register` decorators run.

    Tolerant of a module file not existing yet (the three axes are developed in
    parallel), so the spine is usable before every module has landed.
    """
    for mod in ("dyn_buddy", "dyn_vet", "dyn_pricing"):
        try:  # pragma: no cover - import guard
            __import__(f"petcare.{mod}")
        except Exception:
            pass


class DynamicsRegistry:
    """The set of *enabled* modules for one world. Empty == every hook a no-op."""

    def __init__(self, modules: dict[str, DynamicsModule] | None = None) -> None:
        self._modules: dict[str, DynamicsModule] = modules or {}

    def get(self, name: str) -> DynamicsModule | None:
        return self._modules.get(name)

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(self._modules)

    # lifecycle dispatch ------------------------------------------------------
    def build(self, world: "World") -> None:
        for m in self._modules.values():
            m.build(world)

    def on_hour(self, world: "World") -> None:
        for m in self._modules.values():
            m.on_hour(world)

    # typed accessors for the read-points ------------------------------------
    def pricing(self) -> PricingQuery | None:
        return self.get(PRICING)  # type: ignore[return-value]

    def vet(self) -> VetQuery | None:
        return self.get(VET)  # type: ignore[return-value]


def build_dynamics(world: "World", enabled: tuple[str, ...]) -> DynamicsRegistry:
    """Instantiate the enabled modules for ``world`` (does not call `build`).

    Unknown names are skipped (a scenario typo degrades gracefully, like
    `build_config`). The caller runs `registry.build(world)` once afterwards.
    """
    if enabled:
        _load_builtin_modules()
    modules: dict[str, DynamicsModule] = {}
    for name in enabled:
        cls = _REGISTRY.get(name)
        if cls is not None:
            modules[name] = cls(world)
    return DynamicsRegistry(modules)


def derive_seed(base_seed: int, name: str) -> int:
    """Stable per-module seed: independent of PYTHONHASHSEED and of draw order."""
    return (base_seed * 2654435761 ^ zlib.crc32(name.encode())) & 0xFFFFFFFF
