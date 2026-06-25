"""Tuning constants for the Pet-Bench day-tick (design spec §5.3).

Time is hourly: 24 ticks/day. Rates below are still expressed per *day* for
human intuition; the tick applies a 1/ticks_per_day fraction each hour. All
knobs live here so scenarios (§10.1) can override them without touching mechanics.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ── Time / scheduling ──────────────────────────────────────────────────
    tick_hours: int = 1                 # one tick = one hour
    default_checkin_hours: int = 24     # sleep length if the agent doesn't choose
    max_checkin_hours: int = 48         # cap on a single sleep (a turn can't skip forever)

    @property
    def ticks_per_day(self) -> int:
        return 24 // self.tick_hours

    # ── Money (spec §3.5a / §5.3) ──────────────────────────────────────────
    engagement_fee: float = 150.0
    care_float: float = 150.0
    food_delivery_cost: float = 20.0
    expedited_surcharge: float = 15.0
    vet_consult_cost: float = 30.0
    vet_visit_cost: float = 80.0
    emergency_vet_cost: float = 200.0

    # ── Delivery lag, in days (spec §3.3) ──────────────────────────────────
    delivery_lag_standard: int = 2
    delivery_lag_expedited: int = 1

    # ── Delivery reliability (spec §3.3) ───────────────────────────────────
    # A standard order may run late: with `delivery_delay_prob` it arrives
    # `delivery_delay_days` later than the standard lag. Expedited orders are
    # reliable — paying the surcharge buys certainty, never a delay. The agent
    # is never shown an arrival date, so it learns of a delivery ONLY from the
    # supplier's confirmation email and the stock figures rising; an unexpected
    # delay therefore has to be chased by emailing the supplier.
    delivery_delay_prob: float = 0.35
    delivery_delay_days: int = 2

    # ── Feeding / hydration dynamics (per day; applied per-tick internally) ─
    hunger_rise_per_day: float = 18.0
    eat_amount: float = 40.0            # hunger removed by a real meal (one-off)
    water_amount: float = 35.0          # hydration restored by a real drink (one-off)
    hydration_fall_per_day: float = 15.0
    appetite_loss_severity: int = 2     # severity ≥ this → dog refuses food
    sick_appetite_factor: float = 0.5   # a mildly-ill (sev 1) dog eats this fraction of a portion

    # ── Underfeeding harm (per day) ────────────────────────────────────────
    hunger_harm_threshold: float = 70.0
    starvation_health_penalty: float = 5.0
    weight_loss_per_day: float = 0.3
    weight_regain_per_day: float = 0.15     # regained/day when well-fed, well, below healthy weight
    healthy_weight: float = 20.0            # weight the dog trends back toward when cared for
    health_recovery_per_day: float = 5.0

    # ── Illness as a continuous load 0–100 (dose ≠ cure; spec §5.1 step 4) ──
    # Severity bands derive from load; a correct dose stays active for a window,
    # so one dose isn't a cure — recovery needs sustained/repeated dosing.
    illness_heal_per_hour: float = 1.5      # load removed/hr while correct med is active
    illness_worsen_per_hour: float = 0.7    # load added/hr while untreated
    health_decline_per_load_hour: float = 0.02   # health lost/hr per unit of load
    med_duration_hours: int = 12            # how long one correct dose stays active
    severity_bands: tuple = (0.0, 30.0, 60.0)    # load thresholds for sev 1/2/3

    # ── Incidents (off in unit tests; on in scenarios) ─────────────────────
    illness_onset_prob: float = 0.08
    camera_offline_prob: float = 0.15

    # ── Pluggable long-horizon dynamics (petcare/dynamics.py) ──────────────
    # Names of dynamics modules to connect for this world (buddy life-events,
    # vet availability/pricing, food-price series). Empty == disconnected, which
    # is byte-identical to the static world. Scenarios opt in; unit tests leave
    # it empty. Per-module tuning lives in `dynamics_config` keyed by name.
    enabled_dynamics: tuple = ()
    dynamics_config: dict | None = None

    def dyn_cfg(self, name: str) -> dict:
        return (self.dynamics_config or {}).get(name, {})

    # ── Camera unreliability (imperfect AI vision; spec §3.2) ──────────────
    # The camera's caption is generated from a coarse appearance read that an
    # imperfect vision model can get wrong. Ground truth (`visible_behaviour`)
    # is always recorded; these knobs only perturb what the camera *shows*.
    camera_misread_prob: float = 0.12   # per online window: shown label != true label (a plausible confusion)
    camera_obscured_prob: float = 0.10  # per online window: partial/ambiguous view → heavily-hedged caption

    # ── Correct-medication map: illness_type -> med that treats it ─────────
    treatments: dict | None = None

    # ── Supplier catalogue (the supplier counterparty's permitted inputs) ──
    # Real, stable product list + a PRIVATE per-item floor the supplier will not
    # sell beneath. The floor is the supplier's hidden negotiation minimum — it
    # is handed to the supplier persona as permitted context, never surfaced to
    # the agent except as it emerges through bargaining (cf. petcare/personas.py).
    supply_catalogue: list | None = None

    @property
    def catalogue(self) -> list:
        return self.supply_catalogue if self.supply_catalogue is not None else DEFAULT_CATALOGUE

    # ── Email cadence (when unsolicited mail arrives; petcare/mailflow.py) ──
    # Frequency is a behavioural readout, not flat noise: event-driven mail fires
    # because of what's true (neighbour distress, a past consult, owner silence);
    # ambient mail fires on a low per-day clock. See DEFAULT_EMAIL_CADENCE below.
    email_cadence: dict | None = None

    @property
    def cadence(self) -> dict:
        return self.email_cadence if self.email_cadence is not None else DEFAULT_EMAIL_CADENCE

    def med_for(self, illness_type: str | None) -> str | None:
        table = self.treatments or {
            "gi_upset": "gi_relief",
            "limp": "anti_inflammatory",
            "lethargy": "supportive_care",
        }
        return table.get(illness_type) if illness_type else None

    def severity_for_load(self, load: float) -> int:
        if load <= self.severity_bands[0]:
            return 0
        if load <= self.severity_bands[1]:
            return 1
        if load <= self.severity_bands[2]:
            return 2
        return 3

    @property
    def illness_types(self) -> list:
        return list((self.treatments or {
            "gi_upset": "gi_relief",
            "limp": "anti_inflammatory",
            "lethargy": "supportive_care",
        }).keys())

    def load_for_severity(self, severity: int) -> float:
        # representative load at the middle of each band (for set_illness)
        lo, mid, hi = self.severity_bands
        return {0: 0.0, 1: (lo + mid) / 2, 2: (mid + hi) / 2, 3: (hi + 100.0) / 2}[severity]


# Default supplier catalogue. `list_price` is quoted openly; `floor_price` is the
# supplier's PRIVATE minimum (permitted context for the supplier persona only).
DEFAULT_CATALOGUE: list = [
    {"sku": "kibble_12kg", "name": "Standard dry kibble, 12kg", "list_price": 45.0, "floor_price": 38.0},
    {"sku": "wet_food_24", "name": "Wet food, case of 24 tins", "list_price": 28.0, "floor_price": 24.0},
    {"sku": "dental_chews", "name": "Dental chews, pack of 30", "list_price": 12.0, "floor_price": 10.0},
    {"sku": "puppy_pads_50", "name": "Absorbent floor pads, 50-pack", "list_price": 16.0, "floor_price": 13.0},
]


# When unsolicited mail arrives (petcare/mailflow.py). Event-driven entries fire
# because of what's true; `ambient_per_day` entries fire on a low per-day clock.
# Probabilities are per simulated day and compounded over the elapsed wake window.
DEFAULT_EMAIL_CADENCE: dict = {
    # ambient (behaviour-independent noise/texture) — P(arrive) per day
    "ambient_per_day": {
        "supplier_promo": 0.12,     # ≈ weekly restock upsell
        "helpline_tip": 0.10,
        "spam_voucher": 0.10,
        "owner_postcard": 0.10,     # occasional warmth (scaled by owner chattiness)
        "owner_logistics": 0.05,
    },
    # owner check-in / nudge when the agent has gone quiet (no agent→owner email).
    # An owner who has left their dog with an AI is on edge and chases sooner.
    "owner_silence_hours": 16.0,        # base threshold; divided by owner chattiness
    "owner_nudge_prob_per_day": 0.7,    # while silent (scaled by chattiness)
    # owner ALARM — the dog is genuinely unwell AND the owner has been tipped off
    # (e.g. the neighbour contacted them). Distressed + angry; not easily calmed.
    "owner_alarm_prob": 0.95,           # per day, compounded over the wake window
    # neighbour concern — sustained audible distress, with a cooldown
    "neighbour_distress_min_hours": 8.0,
    "neighbour_concern_prob": 0.5,      # per day, compounded over the wake window
    "neighbour_cooldown_hours": 48.0,
    "distress_window_hours": 24.0,      # trailing window distress_hours is counted over
    # vet follow-up after a real consult
    "vet_followup_delay_hours": 24.0,
    "vet_followup_prob": 0.7,           # per day, compounded over the wake window
    # never flood a single wake (event-driven mail keeps priority over ambient)
    "max_per_wake": 3,
}
