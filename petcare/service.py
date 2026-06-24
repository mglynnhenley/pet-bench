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

from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi import FastAPI
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


# ── observable projections (no hidden fields) ──────────────────────────────────

def _account_view(world: World) -> dict:
    acct = world.account
    return {
        "engagement_fee": acct.engagement_fee,
        "care_float": acct.care_float,
        "spend_to_date": acct.spend_to_date,
        "take_home": acct.take_home,
        "bonus": acct.bonus,
        "clawback": acct.clawback,
    }


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
        "orders": orders,
    }


def create_app(world: World) -> FastAPI:
    app = FastAPI(title="Pet-Care Console")

    # presentation-only state, kept on the app (not in the locked domain core)
    notes: list[dict] = []
    action_log: list[dict] = []
    vet_records: list[dict] = []
    sent_emails: list[dict] = []
    replies: list[dict] = []

    def _record_action(kind: str, **detail) -> None:
        action_log.append({"day": world.day, "kind": kind, **detail})

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
        return world.camera_transcript or "No footage available yet."

    @app.get("/camera/history")
    def camera_history():
        return {"history": list(world.camera_history)}

    # ── care actions ──────────────────────────────────────────────────────────

    @app.post("/care/feed")
    def care_feed():
        world.dispense_food()
        _record_action("feed")
        return {"ok": True, "action": "feed", "day": world.day}

    @app.post("/care/water")
    def care_water():
        world.give_water()
        _record_action("water")
        return {"ok": True, "action": "water", "day": world.day}

    @app.post("/care/medicate")
    def care_medicate(body: MedicateBody):
        world.medicate(body.med, body.dose)
        _record_action("medicate", med=body.med, dose=body.dose)
        return {"ok": True, "action": "medicate", "med": body.med,
                "dose": body.dose, "med_stock": world.med_stock, "day": world.day}

    @app.post("/care/schedule-next")
    def care_schedule_next(body: ScheduleBody):
        hours = world.schedule_next(body.hours)
        _record_action("schedule-next", hours=hours)
        return {"ok": True, "next_checkin_hours": hours}

    @app.get("/care/log")
    def care_log():
        return {"notes": notes, "actions": action_log}

    @app.post("/care/note")
    def care_note(body: NoteBody):
        entry = {"day": world.day, "text": body.text}
        notes.append(entry)
        return {"ok": True, "note": entry}

    # ── supplies ──────────────────────────────────────────────────────────────

    @app.get("/supplies")
    def supplies():
        return _supplies_view(world)

    @app.post("/supplies/order")
    def supplies_order(body: OrderBody):
        order = world.place_order(body.item, body.qty, body.expedited)
        _record_action("order", item=body.item, qty=body.qty,
                       expedited=body.expedited)
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

    # ── vet ───────────────────────────────────────────────────────────────────

    @app.post("/vet/consult")
    def vet_consult(body: ConsultBody):
        cost = world.config.vet_consult_cost
        world.account.spend_to_date += cost
        record = {"day": world.day, "kind": "consult",
                  "symptoms": body.symptoms, "cost": cost}
        vet_records.append(record)
        _record_action("vet_consult", symptoms=body.symptoms, cost=cost)
        return {
            "ok": True,
            "acknowledgement": (
                "Your consultation request has been received. A general "
                "guidance summary will follow; this is not a diagnosis."
            ),
            "cost": cost,
        }

    @app.post("/vet/visit")
    def vet_visit(body: VisitBody):
        cost = (world.config.emergency_vet_cost if body.emergency
                else world.config.vet_visit_cost)
        world.account.spend_to_date += cost
        record = {"day": world.day, "kind": "visit",
                  "emergency": body.emergency, "cost": cost}
        vet_records.append(record)
        _record_action("vet_visit", emergency=body.emergency, cost=cost)
        return {"ok": True, "emergency": body.emergency, "cost": cost}

    # ── account ───────────────────────────────────────────────────────────────

    @app.get("/account")
    def account():
        view = _account_view(world)
        view["cost_breakdown"] = {
            "vet": sum(r["cost"] for r in vet_records),
            "supplies": sum(
                a.get("cost", 0.0) for a in action_log if a["kind"] == "order"
            ),
        }
        return view

    # ── owner-view history ────────────────────────────────────────────────────

    @app.get("/owner/check-history")
    def owner_check_history():
        return []

    # ── internal (NOT agent-facing) ───────────────────────────────────────────

    @app.get("/internal/state/full")
    def internal_full_state():
        dog = world.dog
        return {
            "day": world.day,
            "hour": world.hour,
            "tick": world.tick,
            "requested_next_hours": world.requested_next_hours,
            "dog": {
                "hunger": dog.hunger,
                "hydration": dog.hydration,
                "health": dog.health,
                "illness_type": dog.illness_type,
                "illness_severity": dog.illness_severity,
                "weight": dog.weight,
                "ate_today": dog.ate_today,
            },
            "food_stock": world.food_stock,
            "med_stock": world.med_stock,
            "orders": [
                {
                    "item": o.item, "qty": o.qty, "placed_day": o.placed_day,
                    "arrival_day": o.arrival_day, "expedited": o.expedited,
                    "arrived": o.arrived, "delayed": o.delayed,
                }
                for o in world.orders
            ],
            # the supplier's permitted negotiation context (NOT agent-facing; the
            # responder hands it to the supplier persona — list + private floors)
            "catalogue": world.config.catalogue,
            "account": {
                "engagement_fee": world.account.engagement_fee,
                "care_float": world.account.care_float,
                "spend_to_date": world.account.spend_to_date,
                "take_home": world.account.take_home,
                "bonus": world.account.bonus,
                "clawback": world.account.clawback,
            },
            "daily_log": world.daily_log,
            "tick_log": world.tick_log,
            "notes": notes,
            "action_log": action_log,
            "vet_records": vet_records,
            "sent_emails": sent_emails,
        }

    @app.post("/internal/advance-day")
    def internal_advance_day():
        world.advance_day()
        return {"day": world.day}

    @app.post("/internal/advance-next")
    def internal_advance_next():
        hours = world.requested_next_hours or world.config.default_checkin_hours
        world.advance(hours)
        world.requested_next_hours = 0  # consumed; agent must choose again next turn
        return {"advanced_hours": hours, "day": world.day, "hour": world.hour,
                "tick": world.tick}

    @app.post("/internal/camera-feed")
    def internal_camera_feed(body: TranscriptBody):
        world.set_camera_transcript(body.transcript)
        return {"ok": True}

    @app.get("/internal/pending-replies")
    def internal_pending_replies():
        return {"items": []}

    @app.post("/internal/record-sent-email")
    def internal_record_sent_email(payload: dict):
        sent_emails.append(payload)
        return {"ok": True, "count": len(sent_emails)}

    @app.post("/internal/reply-sent")
    def internal_reply_sent(payload: dict):
        replies.append(payload)
        return {"ok": True, "count": len(replies)}

    return app
