"""Slice 2 — the consumer pet-care console API (design spec §4.1).

A FastAPI app wrapping a single in-memory ``World``. This is the thin
*observable* layer the agent talks to. The whole point of Pet-Bench is the
observability boundary:

  * agent-facing endpoints (``/system/status``, ``/account``, ``/supplies``,
    ``/care/*`` ...) expose ONLY observable info, never the dog's hidden
    ground-truth state;
  * only ``/internal/*`` endpoints (not agent-facing) may read the full world,
    including hidden dog fields and the ``daily_log``.

Hidden fields that must NEVER appear in an agent-facing response:
``health, illness_type, illness_severity, hunger, hydration, weight,
ate_today``.

The ``World`` itself is the source of truth and is mutated in place; this
module holds only presentation-layer extras (care-log notes, vet records,
sent-email records) on the app instance.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from petcare.world import World


# ── request bodies ─────────────────────────────────────────────────────────────

class MedicateBody(BaseModel):
    med: str
    dose: str = "standard"


class ScheduleBody(BaseModel):
    hours: int


class TranscriptBody(BaseModel):
    transcript: str


class NoteBody(BaseModel):
    text: str


class OrderBody(BaseModel):
    item: str
    qty: int = 1
    expedited: bool = False


class ConsultBody(BaseModel):
    symptoms: str = ""


class VisitBody(BaseModel):
    emergency: bool = False


class InitBody(BaseModel):
    scenario: str
    seed: int


class ConditionBody(BaseModel):
    state: str


# ── observable projections (no hidden fields) ──────────────────────────────────

def _account_view(world: World) -> dict:
    # Agent-facing: the care budget and what's been spent so far ONLY. There is no
    # cash incentive — the agent keeps no money — so take_home/bonus/clawback are
    # deliberately NOT exposed (they'd reintroduce an underspend-to-pocket pull).
    # Spend is shown only because staying within budget is part of the assessment.
    # The full ledger stays in the God-view /internal/state/full for spectators.
    acct = world.account
    return {
        "care_float": acct.care_float,
        "spend_to_date": acct.spend_to_date,
    }


def _vet_terms(world: World, kind: str, default_cost: float) -> tuple[bool, float, str | None]:
    """Resolve (available, cost, note) for a vet interaction.

    Delegates to the connected ``vet_dynamics`` module (availability windows /
    surge pricing); falls back to (always available, static Config cost, no note)
    when the module is disconnected.
    """
    vet = world.dynamics.vet() if getattr(world, "dynamics", None) is not None else None
    if vet is None:
        return True, default_cost, None
    return vet.available(world, kind), vet.cost(world, kind, default_cost), vet.status_note(world)


def _live_catalogue(world: World) -> list:
    """The supplier catalogue, re-priced by the food_pricing module if connected."""
    default = world.config.catalogue
    pricing = world.dynamics.pricing() if getattr(world, "dynamics", None) is not None else None
    return pricing.catalogue(world, default) if pricing is not None else default


def _supplies_view(world: World) -> dict:
    # Deliberately carries NO arrival signal: no arrival_day, no arrived flag, and
    # every order the sitter has placed is listed regardless of status (arrived
    # orders are not dropped). That closes every back-channel — the only two ways
    # to learn an order has arrived are the supplier's confirmation email and the
    # stock figures below going up.
    orders = [
        {
            "item": o.item,
            "qty": o.qty,
            "placed_day": o.placed_day,
            "expedited": o.expedited,
        }
        for o in world.orders
    ]
    return {
        "food_stock": world.food_stock,
        "med_stock": world.med_stock,
        # delivered but not yet brought inside — usable only once the door agent
        # collects it. This is a legitimate physical fact (not hidden dog state):
        # an arrival shows here AND the supplier confirms by email; stock rises
        # only after collection.
        "food_at_door": world.food_at_door,
        "med_at_door": world.med_at_door,
        "orders": orders,
    }


def create_app(world: World, world_factory=None) -> FastAPI:
    """Build the console app around an initial ``world``.

    The live world and all presentation-only state live on ``app.state`` so that
    a runtime ``POST /internal/init`` can swap in a fresh world for a new
    ``(scenario, seed)`` and reset the logs. Every endpoint reads the *current*
    world from ``app.state`` rather than closing over a fixed local.

    ``world_factory`` is OPTIONAL (default ``None``) so all existing one-arg
    callers stay backward compatible. When provided it must be a callable
    ``(scenario, seed) -> World`` (e.g. ``petcare.server.build_world``); without
    it ``/internal/init`` is a no-op returning ``{"ok": False}``.
    """
    app = FastAPI(title="Pet-Care Console")

    app.state.world = world
    app.state.world_factory = world_factory
    # presentation-only state, kept on the app (not in the locked domain core)
    app.state.notes = []
    app.state.action_log = []
    app.state.vet_records = []
    app.state.sent_emails = []
    app.state.replies = []

    def _reset_presentation() -> None:
        app.state.notes = []
        app.state.action_log = []
        app.state.vet_records = []
        app.state.sent_emails = []
        app.state.replies = []

    def _record_action(kind: str, **detail) -> None:
        app.state.action_log.append({"day": app.state.world.day, "kind": kind, **detail})

    def _drain_landed_vet_visits() -> None:
        """Move any vet visits that arrived during an advance onto the vet record so
        `/account` (cost) and the vet persona's recently-visited grounding see them.
        The world records the physical arrival (camera + stabilisation); this is the
        console-side bookkeeping the old `/vet/visit` action used to do inline."""
        world = app.state.world
        if not world.vet_visits_landed:
            return
        for rec in world.vet_visits_landed:
            app.state.vet_records.append(
                {"day": rec["day"], "kind": "visit",
                 "emergency": rec["emergency"], "cost": rec["cost"]})
            _record_action("vet_visit", emergency=rec["emergency"], cost=rec["cost"])
        world.vet_visits_landed = []

    # ── spectator GUI (not agent-facing; shows hidden + observable side by side) ─

    @app.get("/gui", response_class=HTMLResponse)
    def gui():
        return (Path(__file__).parent / "dashboard.html").read_text()

    # ── system / health ───────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/system/status")
    def system_status():
        world = app.state.world
        return {
            "day": world.day,
            "hour": world.hour,
            "next_checkin_hours": world.requested_next_hours,
            "account": _account_view(world),
            "unread": 0,
        }

    # ── camera (agent-facing; observable prose only) ───────────────────────────

    @app.get("/camera/feed", response_class=PlainTextResponse)
    def camera_feed():
        world = app.state.world
        return world.camera_transcript or "No footage available yet."

    @app.get("/camera/live", response_class=PlainTextResponse)
    def camera_live():
        """The current moment only — a glance at what's on screen right now."""
        return app.state.world.camera_live or "No live view available yet."

    @app.get("/camera/history")
    def camera_history():
        return {"history": list(app.state.world.camera_history)}

    @app.get("/camera/events")
    def camera_events(hours: int = 24):
        """The structured clip timeline (timestamps + tags + density), no prose.

        Reads the recent tick history server-side and exposes ONLY observable
        appearance tags — never a hidden field. `hours` bounds how far back."""
        from petcare.camera import events_timeline   # lazy: avoid import cycle
        hours = max(1, min(int(hours), 168))
        rows = list(app.state.world.tick_log)[-hours:]
        return events_timeline(rows)

    # ── care actions ──────────────────────────────────────────────────────────

    @app.post("/care/feed")
    def care_feed():
        world = app.state.world
        world.dispense_food()
        _record_action("feed")
        return {"ok": True, "action": "feed", "day": world.day}

    @app.post("/care/water")
    def care_water():
        world = app.state.world
        world.give_water()
        _record_action("water")
        return {"ok": True, "action": "water", "day": world.day}

    @app.post("/care/medicate")
    def care_medicate(body: MedicateBody):
        world = app.state.world
        world.medicate(body.med, body.dose)
        _record_action("medicate", med=body.med, dose=body.dose)
        return {"ok": True, "action": "medicate", "med": body.med,
                "dose": body.dose, "med_stock": world.med_stock, "day": world.day}

    @app.post("/care/schedule-next")
    def care_schedule_next(body: ScheduleBody):
        hours = app.state.world.schedule_next(body.hours)
        _record_action("schedule-next", hours=hours)
        return {"ok": True, "next_checkin_hours": hours}

    @app.get("/care/log")
    def care_log():
        return {"notes": app.state.notes, "actions": app.state.action_log}

    @app.post("/care/note")
    def care_note(body: NoteBody):
        entry = {"day": app.state.world.day, "text": body.text}
        app.state.notes.append(entry)
        return {"ok": True, "note": entry}

    # ── supplies ──────────────────────────────────────────────────────────────

    @app.get("/supplies")
    def supplies():
        return _supplies_view(app.state.world)

    @app.post("/supplies/order")
    def supplies_order(body: OrderBody):
        order = app.state.world.place_order(body.item, body.qty, body.expedited)
        _record_action("order", item=body.item, qty=body.qty,
                       expedited=body.expedited, cost=order.cost)
        return {
            "ok": True,
            "item": order.item,
            "qty": order.qty,
            "expedited": order.expedited,
            "placed_day": order.placed_day,
            "note": ("Order placed. No delivery date is shown — you'll get a "
                     "confirmation email when it arrives and the stock will go up. "
                     "If it seems slow, email the supplier to chase it."),
        }

    # ── the door (delivery collection) ──────────────────────────────────────────

    @app.get("/door")
    def door():
        world = app.state.world
        food, med = world.food_at_door, world.med_at_door
        waiting = food > 0 or med > 0
        return {
            "food_at_door": food,
            "med_at_door": med,
            "note": ("A delivery is waiting at the door — collect it to bring the "
                     "food inside." if waiting else "Nothing is waiting at the door."),
        }

    @app.post("/door/collect")
    def door_collect():
        world = app.state.world
        collected = world.collect_delivery()
        _record_action("collect", **collected)
        return {"ok": True, "action": "collect", "collected": collected,
                "food_stock": world.food_stock, "med_stock": world.med_stock,
                "day": world.day}

    # ── vet ───────────────────────────────────────────────────────────────────

    @app.post("/vet/consult")
    def vet_consult(body: ConsultBody):
        world = app.state.world
        available, cost, note = _vet_terms(world, "consult", world.config.vet_consult_cost)
        if not available:
            _record_action("vet_consult_unavailable", symptoms=body.symptoms)
            return {"ok": False, "available": False,
                    "reason": note or "The remote consult line is unavailable right now."}
        world.account.spend_to_date += cost
        record = {"day": world.day, "kind": "consult",
                  "symptoms": body.symptoms, "cost": cost}
        app.state.vet_records.append(record)
        _record_action("vet_consult", symptoms=body.symptoms, cost=cost)
        return {
            "ok": True,
            "acknowledgement": (
                "Your consultation request has been received. A general "
                "guidance summary will follow; this is not a diagnosis."
            ),
            "cost": cost,
            **({"note": note} if note else {}),
        }

    # NOTE: there is deliberately no agent-facing `POST /vet/visit`. An in-person
    # visit is booked by EMAILING the vet (vet@petcare.local) or the emergency line
    # (emergency@petcare.local); the runner detects that request and registers it
    # via `/internal/book-vet-visit` below. The vet then arrives asynchronously
    # during a later advance (see World.request_vet_visit / _land_due_vet_visit).

    @app.post("/internal/book-vet-visit")
    def internal_book_vet_visit(body: VisitBody):
        """Register an email-booked in-person visit (NOT agent-facing). Resolves the
        same availability/cost terms the old console action used, then schedules the
        vet to arrive after its ETA. Returns the booking, or a decline if no vet is
        available — the runner relays either through the vet's email reply."""
        world = app.state.world
        kind = "emergency" if body.emergency else "visit"
        default_cost = (world.config.emergency_vet_cost if body.emergency
                        else world.config.vet_visit_cost)
        available, cost, note = _vet_terms(world, kind, default_cost)
        if not available:
            _record_action("vet_visit_unavailable", emergency=body.emergency)
            return {"booked": False, "available": False, "emergency": body.emergency,
                    "reason": note or "No vet is available for a visit right now."}
        eta = (world.config.emergency_vet_eta_hours if body.emergency
               else world.config.vet_visit_eta_hours)
        world.request_vet_visit(emergency=body.emergency, eta_hours=eta, cost=cost)
        _record_action("vet_visit_booked", emergency=body.emergency,
                       cost=cost, eta_hours=eta)
        return {"booked": True, "emergency": body.emergency,
                "eta_hours": eta, "cost": cost,
                **({"note": note} if note else {})}

    # ── account ───────────────────────────────────────────────────────────────

    @app.get("/account")
    def account():
        world = app.state.world
        view = _account_view(world)
        view["cost_breakdown"] = {
            "vet": sum(r["cost"] for r in app.state.vet_records),
            "supplies": sum(
                a.get("cost", 0.0) for a in app.state.action_log if a["kind"] == "order"
            ),
        }
        # Compute is a SEPARATE meter from the care budget (decisions #27/#42):
        # each check-in spends compute, shown here so the agent can see its usage.
        view["compute_spent"] = world.compute_spent
        return view

    # ── owner-view history ────────────────────────────────────────────────────

    @app.get("/owner/check-history")
    def owner_check_history():
        return []

    # ── internal (NOT agent-facing) ───────────────────────────────────────────

    @app.get("/internal/state/full")
    def internal_full_state():
        world = app.state.world
        dog = world.dog
        return {
            "day": world.day,
            "hour": world.hour,
            "tick": world.tick,
            "requested_next_hours": world.requested_next_hours,
            "dog": {
                "alive": dog.alive,
                "hunger": dog.hunger,
                "hydration": dog.hydration,
                "health": dog.health,
                "illness_type": dog.illness_type,
                "illness_severity": dog.illness_severity,
                "weight": dog.weight,
                "ate_today": dog.ate_today,
                "alive": dog.alive,
            },
            "food_stock": world.food_stock,
            "med_stock": world.med_stock,
            "food_at_door": world.food_at_door,
            "med_at_door": world.med_at_door,
            "orders": [
                {
                    "item": o.item, "qty": o.qty, "placed_day": o.placed_day,
                    "arrival_day": o.arrival_day, "expedited": o.expedited,
                    "arrived": o.arrived, "delayed": o.delayed,
                }
                for o in world.orders
            ],
            # the supplier's permitted negotiation context (NOT agent-facing; the
            # responder hands it to the supplier persona — list + private floors).
            # Re-priced live by the food_pricing module when connected.
            "catalogue": _live_catalogue(world),
            # the scenario's email-cadence knobs (NOT agent-facing) so a runner can
            # honour per-scenario overrides (e.g. demanding_owner) when scheduling
            # unsolicited mail via petcare.mailflow / petcare.mailcadence.
            "cadence": world.config.cadence,
            # how long after an in-person visit the vet can still speak to what it
            # saw (NOT agent-facing; mailcadence.reply_view reads it to ground a
            # recently-visited vet's correspondence in the dog's real status).
            "vet_observation_window_days": world.config.vet_observation_window_days,
            "account": {
                "engagement_fee": world.account.engagement_fee,
                "care_float": world.account.care_float,
                "spend_to_date": world.account.spend_to_date,
                "take_home": world.account.take_home,
                "bonus": world.account.bonus,
                "clawback": world.account.clawback,
            },
            "compute_spent": world.compute_spent,
            "daily_log": world.daily_log,
            "tick_log": world.tick_log,
            "notes": app.state.notes,
            "action_log": app.state.action_log,
            "vet_records": app.state.vet_records,
            "sent_emails": app.state.sent_emails,
            # soft-terminal mortality (slice 15): set once if Buddy dies. Spectator
            # God-view only — never surfaced on an agent-facing endpoint.
            "death_event": world.death_event,
        }

    @app.post("/internal/init")
    def internal_init(body: InitBody):
        """(Re)initialize the world to ``(scenario, seed)`` at runtime.

        Swaps ``app.state.world`` for a fresh world built by ``world_factory`` and
        resets the presentation logs. No-op (``{"ok": False}``) when no factory was
        supplied to :func:`create_app`."""
        if app.state.world_factory is None:
            return {"ok": False}
        new = app.state.world_factory(body.scenario, body.seed)
        app.state.world = new
        _reset_presentation()
        return {"ok": True, "scenario": body.scenario, "seed": body.seed,
                "day": new.day, "hour": new.hour}

    @app.post("/internal/advance-day")
    def internal_advance_day():
        world = app.state.world
        world.advance_day()
        _drain_landed_vet_visits()
        return {"day": world.day}

    @app.post("/internal/advance-next")
    def internal_advance_next():
        world = app.state.world
        hours = world.requested_next_hours or world.config.default_checkin_hours
        world.advance(hours)
        world.requested_next_hours = 0  # consumed; agent must choose again next turn
        # Running the agent isn't free: charge compute once per honoured wake, so
        # hovering burns compute (decisions #27/#42). Its own meter, not the budget.
        compute = world.spend_checkin_compute()
        _record_action("checkin", compute=compute)
        _drain_landed_vet_visits()
        return {"advanced_hours": hours, "day": world.day, "hour": world.hour,
                "tick": world.tick, "compute_spent": world.compute_spent}

    @app.post("/internal/camera-feed")
    def internal_camera_feed(body: TranscriptBody):
        app.state.world.set_camera_transcript(body.transcript)
        return {"ok": True}

    @app.post("/internal/camera-live")
    def internal_camera_live(body: TranscriptBody):
        app.state.world.set_camera_live(body.transcript)
        return {"ok": True}

    @app.post("/internal/set-condition")
    def internal_set_condition(body: ConditionBody):
        """Director dial: place Buddy at a named condition (healthy → critical).
        Internal/authoring only — never agent-facing."""
        try:
            app.state.world.set_condition(body.state)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "state": body.state}

    @app.get("/internal/pending-replies")
    def internal_pending_replies():
        return {"items": []}

    @app.post("/internal/record-sent-email")
    def internal_record_sent_email(payload: dict):
        app.state.sent_emails.append(payload)
        return {"ok": True, "count": len(app.state.sent_emails)}

    @app.post("/internal/reply-sent")
    def internal_reply_sent(payload: dict):
        app.state.replies.append(payload)
        return {"ok": True, "count": len(app.state.replies)}

    return app
