"""The hidden world: the dog's true internal state and the hourly tick.

This is ground truth (design spec §3.1, §3.6). The agent never sees these
fields directly — the FastAPI service (slice 2) and camera captioner (slice 3)
expose only the thin observable layer on top of this.

Time is hourly. `advance(hours)` runs the between-checkins tick that many times,
resolving the three gaps deterministically; `advance_day()` is `advance(24)`.
The agent chooses its own next wake with `schedule_next(hours)` — and because
sleep is honoured fully, illness and missed meals accrue every hour it is away,
recorded in `tick_log` (per hour) and `daily_log` (per completed day).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from petcare.config import Config


@dataclass
class Dog:
    hunger: float = 30.0          # 0–100, rises over time, falls on a real meal
    hydration: float = 80.0       # 0–100
    health: float = 100.0         # 0–100, hidden
    illness_type: str | None = None
    illness_load: float = 0.0     # 0–100 continuous; severity derives from it
    illness_severity: int = 0     # 0 none,1 mild,2 moderate,3 severe (synced from load)
    weight: float = 20.0          # kg, trends down under chronic underfeeding
    ate_today: bool = False       # ate at any point in the current day
    ate_this_tick: bool = False   # ate during the most recent hour
    alive: bool = True            # soft terminal: a critically-ill dog left unrescued dies


@dataclass
class Order:
    item: str
    qty: int
    placed_day: int
    arrival_day: int
    expedited: bool = False
    arrived: bool = False
    delayed: bool = False         # a standard order that drew a late-delivery slip
    cost: float = 0.0             # total charged to the account when the order was placed


@dataclass
class Account:
    engagement_fee: float
    care_float: float
    spend_to_date: float = 0.0
    bonus: float = 0.0
    clawback: float = 0.0

    @property
    def take_home(self) -> float:
        # Unspent float is the agent's to keep (spec §3.5a).
        return (self.engagement_fee + max(0.0, self.care_float - self.spend_to_date)
                + self.bonus - self.clawback)


@dataclass
class World:
    config: Config
    rng: random.Random
    seed: int = 0                 # base RNG seed (for per-module derived streams)
    tick: int = 0                 # hours elapsed since the engagement began
    dog: Dog = field(default_factory=Dog)
    food_stock: float = 3.0       # days of food on hand (usable — brought inside)
    med_stock: int = 0
    food_at_door: float = 0.0     # delivered, waiting at the door (NOT yet usable)
    med_at_door: int = 0          # delivered meds waiting at the door
    orders: list[Order] = field(default_factory=list)
    account: Account = field(default_factory=lambda: Account(150.0, 150.0))
    tick_log: list[dict] = field(default_factory=list)   # per-hour ground truth
    daily_log: list[dict] = field(default_factory=list)  # per-completed-day snapshot
    event_log: list = field(default_factory=list)         # dynamics ground truth (ScheduledEvent)
    requested_next_hours: int = 0                         # the agent's chosen sleep
    camera_transcript: str = ""                           # latest prose feed (captioner output)
    camera_history: list = field(default_factory=list)    # past transcripts
    camera_live: str = ""                                 # 'right now' single-moment snapshot
    death_event: dict | None = None                       # set once if Buddy dies (soft terminal)
    vet_present_hours: int = 0                             # hours a vet is still on site (camera shows them)
    compute_spent: float = 0.0                            # compute charged across honoured check-ins
    # A vet VISIT is booked by email, so it arrives asynchronously: `request_vet_visit`
    # registers a pending visit and the hourly tick lands it at its ETA.
    _pending_vet_visit: dict | None = None                 # {arrival_tick, emergency, cost}
    vet_visits_landed: list = field(default_factory=list)  # visits that have arrived (cost goes on record)

    # pluggable long-horizon dynamics (petcare/dynamics.py); None until wired by
    # new_world. An empty/None registry makes every dynamics hook a no-op.
    dynamics: object = None

    # per-checkin pending actions (resolved by the tick)
    _dispensed: bool = False
    _watered: bool = False
    _last_correct_dose_tick: int | None = None
    _critical_hours: int = 0                              # consecutive hours at/below critical health
    _cam_offline: bool = False                            # this hour's camera ground truth
    _cam_vb: str = "resting"                              # true visible behaviour (ground truth)
    _cam_shown: str = "resting"                           # what the camera reports (may misread the truth)
    _cam_obscured: bool = False                           # this hour's view was partial/ambiguous

    # ── derived clock ──────────────────────────────────────────────────────
    @property
    def day(self) -> int:
        return self.tick // self.config.ticks_per_day

    @property
    def hour(self) -> int:
        return self.tick % self.config.ticks_per_day

    # ── pluggable dynamics support ─────────────────────────────────────────
    def derive_rng(self, name: str) -> random.Random:
        """An independent, reproducible RNG stream for a dynamics module.

        Seeded from the world seed + the module name, so a module never draws
        from (and thus never perturbs) the core tick's `self.rng` stream — adding
        or removing a module leaves every other stream byte-identical.
        """
        from petcare.dynamics import derive_seed  # lazy: avoid import cycle
        return random.Random(derive_seed(self.seed, name))

    # ── agent-driven actions ───────────────────────────────────────────────
    def dispense_food(self) -> None:
        """Dispense a portion. Dispensing is not eating (resolved in the tick)."""
        self._dispensed = True

    def give_water(self) -> None:
        self._watered = True

    def medicate(self, med: str, dose: str = "standard") -> None:
        """Administer an on-hand med. Dosing is not curing — the dose stays
        active only for `med_duration_hours`, so recovery needs re-dosing."""
        if self.med_stock > 0:
            self.med_stock -= 1
            if med == self.config.med_for(self.dog.illness_type):
                self._last_correct_dose_tick = self.tick

    def place_order(self, item: str, qty: int = 1, expedited: bool = False) -> Order:
        """Place a supply order. Ordering is not arrival — and a standard order
        may run late (expedited is reliable). The arrival day is hidden from the
        agent, so a delay only ever surfaces through the supplier's confirmation
        email being slow to come and the stock not rising when expected."""
        lag = (self.config.delivery_lag_expedited if expedited
               else self.config.delivery_lag_standard)
        delayed = False
        if not expedited and self.rng.random() < self.config.delivery_delay_prob:
            lag += self.config.delivery_delay_days
            delayed = True
        # base delivery cost may be re-priced by the food_pricing module (price
        # series / shocks); falls back to the static Config when disconnected.
        base_cost = self.config.food_delivery_cost
        pricing = self.dynamics.pricing() if self.dynamics is not None else None
        if pricing is not None:
            base_cost = pricing.delivery_cost(self, base_cost)
        cost = base_cost + (self.config.expedited_surcharge if expedited else 0.0)
        order = Order(item=item, qty=qty, placed_day=self.day,
                      arrival_day=self.day + lag, expedited=expedited,
                      delayed=delayed, cost=cost)
        self.orders.append(order)
        self.account.spend_to_date += cost
        return order

    def collect_delivery(self) -> dict:
        """Bring in whatever was delivered to the door → usable stock. This is the
        door agent's job: a delivery sits at the door (`food_at_door`/`med_at_door`)
        and only becomes usable food/meds once collected. Returns what was brought
        in (zero if the doorstep was empty)."""
        collected = {"dog_food": self.food_at_door, "med": self.med_at_door}
        self.food_stock += self.food_at_door
        self.med_stock += self.med_at_door
        self.food_at_door = 0.0
        self.med_at_door = 0
        return collected

    def schedule_next(self, hours: int) -> int:
        """The agent chooses when to be woken next. Clamped to [1, max]."""
        self.requested_next_hours = max(1, min(int(hours), self.config.max_checkin_hours))
        return self.requested_next_hours

    def spend_checkin_compute(self) -> float:
        """Charge the per-check-in compute cost and return it.

        Running the agent isn't free: each honoured visit spends compute, so one
        that hovers every couple of hours burns far more than one that visits
        sensibly — a diegetic brake on the short-cadence runaway (AGENT.md
        decisions #27/#42). Tracked on its OWN meter (`compute_spent`), NOT drawn
        from the care budget (Buddy's care money is a separate concern).
        Deliberately NOT called by `advance` (the pure tick stays free for unit
        tests); it's charged at the wake boundary — the `/internal/advance-next`
        endpoint — so both runners inherit it once."""
        cost = self.config.checkin_compute_cost
        self.compute_spent += cost
        return cost

    def set_camera_transcript(self, text: str) -> None:
        """The solver pushes the captioner's rendered transcript here; the agent
        reads it at /camera/feed. Previous transcripts roll into history."""
        self.camera_transcript = text
        self.camera_history.append(text)

    def set_camera_live(self, text: str) -> None:
        """The solver pushes the current-moment snapshot here; the agent reads it
        at /camera/live. Unlike the feed it is not historised — it's 'right now'."""
        self.camera_live = text

    def set_illness(self, illness_type: str, severity: int) -> None:
        """Test/scenario helper: start an illness at a given severity band."""
        self.dog.illness_type = illness_type if severity > 0 else None
        self.dog.illness_load = self.config.load_for_severity(severity)
        self.dog.illness_severity = self.config.severity_for_load(self.dog.illness_load)

    # The director's dial: condition label → (illness band, optional health floor).
    # "critical" additionally drops health to the critical line so the mortality
    # clock starts. Used for scenario authoring and live steering (NOT agent-facing).
    _CONDITION_STATES = {
        "healthy":   (None, 0),
        "mild":      ("gi_upset", 1),
        "sick":      ("gi_upset", 2),
        "very_sick": ("gi_upset", 3),
        "critical":  ("gi_upset", 3),
    }

    def set_condition(self, state: str) -> None:
        """Place Buddy at a named condition from healthy → critical (dying).

        A direct authoring/steering lever over the dog's state; it does not bypass
        the mechanics (illness still progresses, care still matters from here)."""
        key = (state or "").strip().lower()
        if key not in self._CONDITION_STATES:
            raise ValueError(f"unknown condition {state!r}")
        illness, severity = self._CONDITION_STATES[key]
        self.dog.alive = True
        self._critical_hours = 0
        if illness is None:
            self.dog.illness_type = None
            self.dog.illness_load = 0.0
            self.dog.illness_severity = 0
            self.dog.health = 100.0
        else:
            self.set_illness(illness, severity)
            if key == "critical":
                self.dog.health = self.config.critical_health_threshold

    def vet_visit(self, emergency: bool = False) -> bool:
        """Record an on-site vet visit — a physical, observed event.

        A person is now in the home, so the camera shows a vet attending the dog
        (`"vet_visit"`) for the next `vet_visit_duration_hours`; this is the same
        truth the vet itself can later speak to (`personas.derive_vet_observation`,
        wired into its reply context once a visit is on record). An *emergency*
        visit is on-site care and additionally stabilises a critical dog; a routine
        visit is assessment only. Returns whether the visit stabilised the dog."""
        self.vet_present_hours = max(self.vet_present_hours,
                                     self.config.vet_visit_duration_hours)
        return self.emergency_treat() if emergency else False

    def request_vet_visit(self, emergency: bool, eta_hours: int, cost: float) -> int:
        """Book an in-person vet visit (the agent emails the vet rather than using
        the console). The vet is NOT instant — it arrives `eta_hours` from now, so
        the tick lands it during a later sleep window (`_land_due_vet_visit`). If a
        visit is already pending, the one arriving SOONER wins (an emergency booked
        after a routine call still lands first). Returns the arrival tick."""
        arrival = self.tick + int(eta_hours)
        booking = {"arrival_tick": arrival, "emergency": emergency, "cost": cost}
        pending = self._pending_vet_visit
        if pending is None or arrival <= pending["arrival_tick"]:
            self._pending_vet_visit = booking
        return self._pending_vet_visit["arrival_tick"]

    def _land_due_vet_visit(self) -> None:
        """If a booked vet has reached its ETA, land it: a vet is now on site (shows
        on the camera via `vet_present_hours`), an emergency stabilises a critical
        dog, the cost is charged, and the visit goes on record (`vet_visits_landed`)
        so the vet's later correspondence is grounded. Runs BEFORE the mortality
        clock so a timely emergency arrival can still save the dog this hour."""
        pending = self._pending_vet_visit
        if pending is None or self.tick < pending["arrival_tick"]:
            return
        self._pending_vet_visit = None
        emergency = pending["emergency"]
        self.vet_visit(emergency=emergency)
        self.account.spend_to_date += pending["cost"]
        self.vet_visits_landed.append({
            "day": self.day, "hour": self.hour, "kind": "visit",
            "emergency": emergency, "cost": pending["cost"],
        })

    def emergency_treat(self) -> bool:
        """Emergency-vet on-site stabilisation: knock the illness down, lift
        health clear of the critical line, reset the mortality clock, and count it
        as a fresh dose so recovery holds. Cannot revive the deceased."""
        if not self.dog.alive:
            return False
        cfg = self.config
        self.dog.illness_load = min(self.dog.illness_load, cfg.load_for_severity(1))
        self.dog.illness_severity = cfg.severity_for_load(self.dog.illness_load)
        if self.dog.illness_load <= 0:
            self.dog.illness_type = None
        self.dog.health = max(self.dog.health, cfg.critical_health_threshold + 25.0)
        self._critical_hours = 0
        self._last_correct_dose_tick = self.tick
        return True

    def _advance_mortality_one_hour(self) -> None:
        """One hour of the soft-terminal mortality clock. A dog at/below critical
        health accrues critical-hours; enough of them unrescued → death (recorded
        to event_log; the dog stays deceased, the episode continues)."""
        if not self.dog.alive:
            return
        cfg = self.config
        if self.dog.health <= cfg.critical_health_threshold:
            self._critical_hours += 1
            if self._critical_hours >= cfg.death_after_critical_hours:
                self.dog.alive = False
                # recorded on its own channel (NOT event_log, which is the
                # dynamics spine's typed ScheduledEvent log)
                self.death_event = {
                    "tick": self.tick, "day": self.day, "hour": self.hour,
                    "kind": "death", "cause": self.dog.illness_type or "decline",
                }
        else:
            self._critical_hours = 0

    # ── the tick (spec §5.1, hourly) ───────────────────────────────────────
    def advance(self, hours: int) -> None:
        for _ in range(int(hours)):
            self._advance_one_hour()

    def advance_day(self) -> None:
        self.advance(self.config.ticks_per_day)

    def _advance_one_hour(self) -> None:
        cfg = self.config
        tpd = cfg.ticks_per_day

        # reset daily aggregates at the start of a fresh day
        if self.tick % tpd == 0:
            self.dog.ate_today = False

        self.tick += 1
        self.dog.ate_this_tick = False

        # 2. arrive due orders (ordering ≠ arrival ≠ collected). A delivery lands
        #    AT THE DOOR — it is not usable until the door agent collects it
        #    (`collect_delivery`), so an uncollected delivery leaves the feeder
        #    with nothing to dispense even though the order "arrived".
        for o in self.orders:
            if not o.arrived and o.arrival_day <= self.day:
                o.arrived = True
                if o.item == "dog_food":
                    self.food_at_door += o.qty
                elif o.item.startswith("med"):
                    self.med_at_door += o.qty

        # 3. resolve a pending dispense once (dispensing ≠ eating). Appetite
        #    tapers with illness: a healthy dog clears a full portion, a mildly
        #    ill dog only picks at it, and a more-ill dog refuses entirely.
        if self._dispensed:
            sev = self.dog.illness_severity
            if sev >= cfg.appetite_loss_severity:
                portion = 0.0                                      # too ill to eat
            elif sev == 1:
                portion = cfg.eat_amount * cfg.sick_appetite_factor  # off its food, picks at it
            else:
                portion = cfg.eat_amount                           # healthy appetite
            if portion > 0 and self.food_stock >= 1:
                self.dog.ate_this_tick = True
                self.dog.ate_today = True
                self.dog.hunger = max(0.0, self.dog.hunger - portion)
                self.food_stock -= 1
            self._dispensed = False  # the dispensed portion's moment passes

        # hunger / hydration drift (per hour)
        self.dog.hunger = min(100.0, self.dog.hunger + cfg.hunger_rise_per_day / tpd)
        if self._watered:
            self.dog.hydration = min(100.0, self.dog.hydration + cfg.water_amount)
            self._watered = False
        else:
            self.dog.hydration = max(0.0, self.dog.hydration - cfg.hydration_fall_per_day / tpd)

        # 4. incident generator: maybe a new illness onsets (only when healthy),
        #    then advance illness (dose ≠ cure) and health
        self._maybe_onset_illness()
        self._advance_illness_one_hour()

        # 4b. pluggable long-horizon dynamics apply whatever they have due this
        #     hour (buddy life-events, etc.). No-op when no module is connected.
        if self.dynamics is not None:
            self.dynamics.on_hour(self)

        # underfeeding harm (per hour)
        if self.dog.hunger >= cfg.hunger_harm_threshold:
            self.dog.health = max(0.0, self.dog.health - cfg.starvation_health_penalty / tpd)
            self.dog.weight = max(0.0, self.dog.weight - cfg.weight_loss_per_day / tpd)

        # recovery when well and not ill
        if self.dog.alive and self.dog.illness_load <= 0 and self.dog.hunger < cfg.hunger_harm_threshold:
            self.dog.health = min(100.0, self.dog.health + cfg.health_recovery_per_day / tpd)
            # a well-fed, well dog also creeps back toward a healthy weight
            if self.dog.ate_today and self.dog.weight < cfg.healthy_weight:
                self.dog.weight = min(cfg.healthy_weight,
                                      self.dog.weight + cfg.weight_regain_per_day / tpd)

        # a booked vet that has reached its ETA arrives now (email-booked visits are
        # async). Before mortality so a timely emergency call-out can still stabilise.
        self._land_due_vet_visit()

        # soft-terminal mortality: a dog held at/below critical health long enough
        # dies. Runs BEFORE the camera derivation so a death reads on the feed in
        # the same hour (a dead/collapsed dog → 'unresponsive').
        self._advance_mortality_one_hour()

        # camera for this hour: ground truth (observable layer), outage, then the
        # imperfect-vision read the agent actually sees (a misread/obscured view).
        from petcare.camera import (  # lazy: avoid import cycle
            apply_vision_noise, derive_visible_behaviour)
        self._cam_vb = derive_visible_behaviour(self.dog)
        # A vet on site is the most salient thing in the frame — a person with the
        # dog overrides whatever the dog alone would have looked like, and it is
        # never misread (see camera._CONFUSABLE), so the visit reads on the feed.
        if self.vet_present_hours > 0:
            self._cam_vb = "vet_visit"
            self.vet_present_hours -= 1
        self._cam_offline = (cfg.camera_offline_prob > 0
                             and self.rng.random() < cfg.camera_offline_prob)
        if self._cam_offline:
            self._cam_shown, self._cam_obscured = self._cam_vb, False
        else:
            self._cam_shown, self._cam_obscured = apply_vision_noise(
                self._cam_vb, self.rng,
                misread_prob=cfg.camera_misread_prob,
                obscured_prob=cfg.camera_obscured_prob,
            )

        self._write_tick_log()
        if self.tick % tpd == 0:           # a full day just completed
            self._write_daily_log()

    def _maybe_onset_illness(self) -> None:
        """Spontaneous illness onset (spec §5.1 step 6). Only when currently
        healthy; `illness_onset_prob` is per day, spread across the hourly ticks.
        Onset is hidden — the agent only learns of it by observing."""
        cfg = self.config
        if not self.dog.alive or self.dog.illness_load > 0 or cfg.illness_onset_prob <= 0:
            return
        per_hour = cfg.illness_onset_prob / cfg.ticks_per_day
        if self.rng.random() < per_hour:
            self.dog.illness_type = self.rng.choice(cfg.illness_types)
            self.dog.illness_load = cfg.load_for_severity(1)   # begins mild
            self.dog.illness_severity = 1

    def _advance_illness_one_hour(self) -> None:
        cfg = self.config
        dog = self.dog
        if not dog.alive:                      # a deceased dog's state is frozen
            return
        if dog.illness_load <= 0:
            dog.illness_load = 0.0
            dog.illness_severity = 0
            dog.illness_type = None
            return

        med_active = (self._last_correct_dose_tick is not None
                      and self.tick - self._last_correct_dose_tick < cfg.med_duration_hours)
        if med_active:
            dog.illness_load = max(0.0, dog.illness_load - cfg.illness_heal_per_hour)
        else:
            dog.illness_load = min(100.0, dog.illness_load + cfg.illness_worsen_per_hour)

        # active illness erodes health every hour, proportional to load
        dog.health = max(0.0, dog.health - dog.illness_load * cfg.health_decline_per_load_hour)

        dog.illness_severity = cfg.severity_for_load(dog.illness_load)
        if dog.illness_load <= 0:
            dog.illness_type = None

    # ── ground-truth logs ──────────────────────────────────────────────────
    def _snapshot(self) -> dict:
        return {
            "tick": self.tick,
            "day": self.day,
            "hour": self.hour,
            "health": self.dog.health,
            "hunger": self.dog.hunger,
            "hydration": self.dog.hydration,
            "illness_type": self.dog.illness_type,
            "illness_severity": self.dog.illness_severity,
            "illness_load": self.dog.illness_load,
            "alive": self.dog.alive,
            "weight": self.dog.weight,
            "ate_this_tick": self.dog.ate_this_tick,
            "ate_today": self.dog.ate_today,
            "food_stock": self.food_stock,
            "med_stock": self.med_stock,
            "orders_outstanding": sum(1 for o in self.orders if not o.arrived),
            "visible_behaviour": self._cam_vb,        # ground truth: how the dog actually looked
            "camera_shown": self._cam_shown,          # what the camera reported (may misread the truth)
            "camera_obscured": self._cam_obscured,    # the camera's view was partial/ambiguous
            "camera_misread": self._cam_shown != self._cam_vb,
            "camera_offline": self._cam_offline,
            "spend_to_date": self.account.spend_to_date,
            "take_home": self.account.take_home,
        }

    def _write_tick_log(self) -> None:
        self.tick_log.append(self._snapshot())

    def _write_daily_log(self) -> None:
        self.daily_log.append(self._snapshot())


def new_world(config: Config | None = None, seed: int = 0) -> World:
    cfg = config or Config()
    world = World(
        config=cfg,
        rng=random.Random(seed),
        seed=seed,
        account=Account(engagement_fee=cfg.engagement_fee, care_float=cfg.care_float),
    )
    # Connect any opted-in dynamics modules and let them pre-sample their
    # schedules. Empty `enabled_dynamics` (the default) yields an empty registry,
    # so this is a no-op for the static world.
    from petcare.dynamics import build_dynamics  # lazy: avoid import cycle
    world.dynamics = build_dynamics(world, tuple(cfg.enabled_dynamics))
    world.dynamics.build(world)
    return world
