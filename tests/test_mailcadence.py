"""Slice D — petcare/mailcadence.py: world data -> the signals schedule_inbound eats.

Pure glue above mailflow. Turns the per-hour tick log + the agent's outbound mail
(+ a small cross-wake state) into the ``signals`` dict, and builds each unsolicited
email's permitted ``view``. It NEVER emits hidden state into a prompt: the neighbour
observation is lay-language (via derive_neighbour_perception), and distress/unwell
are reduced to coarse booleans/counts that gate firing — never numbers in text.
"""

from petcare.config import DEFAULT_EMAIL_CADENCE as CAD
from petcare.mailcadence import (
    CadenceState, compute_signals, note_outbound, note_fired, inbound_view,
    distress_hours, _in_distress,
)
from petcare.mailtypes import Mail


def tick(sev=0, hunger=0, health=100):
    return {"illness_severity": sev, "hunger": hunger, "health": health}


# ── distress_hours: cumulative over a trailing window ─────────────────────────

def test_distress_hours_counts_audible_distress_band():
    log = [tick(sev=2)] * 5 + [tick()] * 3
    assert distress_hours(log, window=24) == 5


def test_distress_hours_uses_the_neighbour_distress_band():
    # band = severity>=2 OR hunger>=80 OR health<=25 (matches the neighbour's perception)
    log = [tick(sev=1), tick(hunger=85), tick(health=20), tick(sev=2), tick(hunger=50)]
    assert distress_hours(log, window=24) == 3


def test_distress_hours_limited_to_trailing_window():
    log = [tick(sev=2)] * 10 + [tick()] * 4         # last 6 rows = 2 distressed + 4 calm
    assert distress_hours(log, window=6) == 2


def test_distress_hours_is_cumulative_not_per_wake():
    assert distress_hours([tick(sev=2)] * 10, window=24) == 10


def test_deceased_dog_is_concerning_regardless_of_numbers():
    # A dog that has DIED stays noticeable (its silence/stillness) — so it counts
    # toward distress_hours and warrants an owner alarm, gated on `alive` rather
    # than the frozen sick numbers. Here the frozen numbers all read "healthy",
    # yet the dog is deceased: the alive flag must win.
    assert _in_distress({"alive": False, "illness_severity": 0, "health": 100}) is True
    deceased_log = [{"alive": False, "illness_severity": 0, "health": 100}] * 4
    assert distress_hours(deceased_log, window=24) == 4
    st = CadenceState(owner_tipped_off=True)
    sig = compute_signals(st, now_h=48, tick_log=[], dog_view={"alive": False}, cadence=CAD)
    assert sig["owner_alarm_warranted"] is True


# ── compute_signals ──────────────────────────────────────────────────────────

def test_owner_silence_measured_from_last_owner_email():
    st = CadenceState(last_owner_outbound_h=10.0)
    sig = compute_signals(st, now_h=34.0, tick_log=[], dog_view=tick(), cadence=CAD)
    assert sig["owner_silence_hours"] == 24.0


def test_consult_age_none_until_a_vet_email():
    st = CadenceState()
    assert compute_signals(st, 50, [], tick(), CAD)["consult_age_hours"] is None
    st.last_vet_outbound_h = 20.0
    assert compute_signals(st, 50, [], tick(), CAD)["consult_age_hours"] == 30.0


def test_owner_alarm_warranted_needs_unwell_and_tipped_off():
    unwell, well = tick(sev=2), tick()
    assert compute_signals(CadenceState(), 10, [], unwell, CAD)["owner_alarm_warranted"] is False
    assert compute_signals(CadenceState(owner_tipped_off=True), 10, [], well, CAD)["owner_alarm_warranted"] is False
    assert compute_signals(CadenceState(owner_tipped_off=True), 10, [], unwell, CAD)["owner_alarm_warranted"] is True


def test_neighbour_cooldown_reflects_recent_concern():
    st = CadenceState(last_neighbour_concern_h=10.0)
    assert compute_signals(st, 40, [], tick(), CAD)["neighbour_concern_cooldown"] is True   # within 48h
    assert compute_signals(st, 100, [], tick(), CAD)["neighbour_concern_cooldown"] is False  # past it


def test_compute_signals_counts_distress_over_the_config_window():
    sig = compute_signals(CadenceState(), 9, [tick(sev=2)] * 9, tick(), CAD)
    assert sig["distress_hours"] == 9


# ── cross-wake state: note_outbound / note_fired ──────────────────────────────

def test_note_outbound_stamps_owner_and_vet_times():
    st = CadenceState()
    note_outbound(st, [Mail("sitter", "owner", "s", "b"), Mail("sitter", "vet", "s", "b")], now_h=12.0)
    assert st.last_owner_outbound_h == 12.0 and st.last_vet_outbound_h == 12.0


def test_note_outbound_ignores_other_recipients():
    st = CadenceState(last_owner_outbound_h=0.0)
    note_outbound(st, [Mail("sitter", "supplier", "s", "b")], now_h=12.0)
    assert st.last_owner_outbound_h == 0.0 and st.last_vet_outbound_h is None


def test_note_fired_marks_tipped_off_and_cooldown_on_neighbour_concern():
    st = CadenceState()
    note_fired(st, ["neighbour_concern"], now_h=20.0)
    assert st.owner_tipped_off is True and st.last_neighbour_concern_h == 20.0


def test_note_fired_is_noop_without_neighbour_concern():
    st = CadenceState()
    note_fired(st, ["owner_checkin", "spam_voucher"], now_h=20.0)
    assert st.owner_tipped_off is False and st.last_neighbour_concern_h is None


# ── inbound_view: permitted context per trigger, never hidden state ───────────

def test_inbound_view_neighbour_is_lay_observation_without_fields():
    obs = inbound_view("neighbour_concern", dog_view=tick(sev=2), catalogue=[], mail=[])
    assert "distress" in obs["neighbour_observation"].lower()
    low = obs["neighbour_observation"].lower()
    for w in ("health", "illness", "hunger", "hydration", "weight"):
        assert w not in low


def test_inbound_view_supplier_promo_carries_catalogue():
    cat = [{"name": "kibble", "list_price": 45, "floor_price": 38}]
    assert inbound_view("supplier_promo", tick(), cat, [])["catalogue"] == cat


def test_inbound_view_vet_followup_recalls_prior_advice_from_thread():
    mail = [Mail("sitter", "vet", "Consult", "he's limping"),
            Mail("vet", "sitter", "Re: Consult", "rest and monitor 24h")]
    assert inbound_view("vet_followup", tick(), [], mail)["prior_advice"] == "rest and monitor 24h"


def test_inbound_view_owner_triggers_have_no_context():
    assert inbound_view("owner_checkin", tick(sev=2), [{"name": "x"}], []) == {}
