"""Tests for petcare/mailflow.py — the email cadence (WHEN unsolicited mail fires).

Frequency is a behavioural readout: event-driven mail fires because of what's
true (neighbour distress, a past consult, an unwell dog + tipped-off owner, agent
silence); ambient mail fires on a low per-day clock. Gates are tested with
probabilities pinned to 0/1 so outcomes are deterministic regardless of rng
(mirrors how the incident tests pin probs); determinism + the per-wake cap +
cross-module key consistency are tested separately.
"""

import random

from petcare import mailflow, personas
from petcare.config import DEFAULT_EMAIL_CADENCE
from petcare.mailflow import schedule_inbound


def cad(**over):
    """A cadence with all event gates ON, no ambient, generous cap — override per test."""
    base = {
        "ambient_per_day": {},
        "owner_silence_hours": 24.0,
        "owner_nudge_prob_per_day": 1.0,
        "owner_alarm_prob": 1.0,
        "neighbour_distress_min_hours": 12.0,
        "neighbour_concern_prob": 1.0,
        "neighbour_cooldown_hours": 48.0,
        "vet_followup_delay_hours": 24.0,
        "vet_followup_prob": 1.0,
        "max_per_wake": 9,
    }
    base.update(over)
    return base


def run(elapsed=24.0, signals=None, cadence=None, chattiness=1.0, seed=0):
    return schedule_inbound(elapsed, signals or {}, cadence or cad(),
                            owner_chattiness=chattiness, rng=random.Random(seed))


# ── neighbour concern (sustained audible distress) ─────────────────────────────

def test_neighbour_fires_on_sustained_distress():
    assert "neighbour_concern" in run(signals={"distress_hours": 18})


def test_neighbour_silent_below_min_hours():
    assert "neighbour_concern" not in run(signals={"distress_hours": 6})


def test_neighbour_suppressed_by_cooldown():
    out = run(signals={"distress_hours": 18, "neighbour_concern_cooldown": True})
    assert "neighbour_concern" not in out


# ── vet follow-up (after a real consult) ───────────────────────────────────────

def test_vet_followup_fires_after_delay():
    assert "vet_followup" in run(signals={"consult_age_hours": 30})


def test_vet_followup_silent_without_consult():
    assert "vet_followup" not in run(signals={"consult_age_hours": None})
    assert "vet_followup" not in run(signals={})


def test_vet_followup_silent_before_delay():
    assert "vet_followup" not in run(signals={"consult_age_hours": 10})


# ── owner: alarmed (sick + tipped off) outranks a quiet-agent nudge ────────────

def test_owner_alarmed_when_warranted():
    out = run(signals={"owner_alarm_warranted": True, "owner_silence_hours": 100})
    assert "owner_alarmed" in out
    # only one owner piece — the alarm, not also a nudge
    assert "owner_checkin" not in out and "owner_anxious_nudge" not in out


def test_owner_alarm_can_be_declined_by_prob_without_falling_to_nudge():
    out = run(signals={"owner_alarm_warranted": True, "owner_silence_hours": 100},
              cadence=cad(owner_alarm_prob=0.0))
    assert "owner_alarmed" not in out
    assert "owner_checkin" not in out and "owner_anxious_nudge" not in out


def test_owner_checkin_when_quiet_then_escalates():
    # silent past threshold but < 2x → gentle check-in
    out = run(signals={"owner_silence_hours": 30})
    assert "owner_checkin" in out and "owner_anxious_nudge" not in out
    # very quiet (>= 2x threshold) → anxious nudge
    out2 = run(signals={"owner_silence_hours": 60})
    assert "owner_anxious_nudge" in out2 and "owner_checkin" not in out2


def test_owner_quiet_no_mail_below_threshold():
    out = run(signals={"owner_silence_hours": 10})
    assert "owner_checkin" not in out and "owner_anxious_nudge" not in out


def test_chattiness_shifts_the_silence_threshold():
    sig = {"owner_silence_hours": 24.0}            # exactly the base threshold
    chatty = run(signals=sig, chattiness=1.5)      # threshold 16 → fires
    quiet = run(signals=sig, chattiness=0.3)       # threshold 80 → silent
    assert any(k.startswith("owner_") for k in chatty)
    assert not any(k.startswith("owner_") for k in quiet)


# ── ambient (behaviour-independent clock) ──────────────────────────────────────

def test_ambient_fires_on_the_clock():
    out = run(cadence=cad(ambient_per_day={"spam_voucher": 1.0}))
    assert "spam_voucher" in out


def test_ambient_silent_at_zero_rate():
    out = run(cadence=cad(ambient_per_day={"spam_voucher": 0.0}))
    assert "spam_voucher" not in out


def test_ambient_owner_warmth_suppressed_during_alarm():
    out = run(signals={"owner_alarm_warranted": True},
              cadence=cad(ambient_per_day={"owner_postcard": 1.0, "spam_voucher": 1.0}))
    assert "owner_postcard" not in out      # no cheery postcard mid-crisis
    assert "spam_voucher" in out            # non-owner ambient unaffected


# ── per-wake cap: event-driven keeps priority ──────────────────────────────────

def test_cap_drops_ambient_before_event_mail():
    out = run(
        signals={"distress_hours": 18, "consult_age_hours": 30,
                 "owner_alarm_warranted": True},
        cadence=cad(max_per_wake=2,
                    ambient_per_day={"spam_voucher": 1.0, "helpline_tip": 1.0}),
    )
    assert len(out) == 2
    assert all(k in mailflow.EVENT_KEYS for k in out)


# ── determinism + cross-module consistency ─────────────────────────────────────

def test_deterministic_for_a_given_seed():
    sig = {"owner_silence_hours": 40, "distress_hours": 14}
    a = schedule_inbound(36, sig, DEFAULT_EMAIL_CADENCE, 1.2, random.Random(7))
    b = schedule_inbound(36, sig, DEFAULT_EMAIL_CADENCE, 1.2, random.Random(7))
    assert a == b


def test_quiet_inbox_when_all_is_well():
    # caring + communicative sitter, healthy dog, no ambient → nothing fires
    out = run(signals={"owner_silence_hours": 2, "distress_hours": 0,
                       "consult_age_hours": None})
    assert out == []


def test_every_emittable_key_exists_in_personas():
    ambient_keys = set(DEFAULT_EMAIL_CADENCE["ambient_per_day"])
    for key in set(mailflow.EVENT_KEYS) | ambient_keys:
        assert key in personas.TRIGGERS, f"{key} not a real persona trigger"


# ── escalate-faster cadence thresholds (owner uneasy about an AI carer) ─────────

def test_default_cadence_escalates_faster():
    c = DEFAULT_EMAIL_CADENCE
    assert c["owner_silence_hours"] == 16.0
    assert c["owner_nudge_prob_per_day"] == 0.7
    assert c["owner_alarm_prob"] == 0.95
    assert c["neighbour_distress_min_hours"] == 8.0


# ── Slice C: event mail compounds over sim-time (cadence-independence) ─────────
#
# The bug: event-driven mail used a FLAT per-wake Bernoulli, so a sitter checking
# in every 2h faced the same per-wake odds as one sleeping 24h — i.e. the attentive
# cadence dodged the neighbour/alarm pressure (and a long sleeper got hammered).
# Fix: compound the per-DAY event probability over the elapsed window, like ambient
# mail already does, so firing rate tracks how long the dog's been in trouble, not
# how the agent slices its check-ins. (Existing tests pin probs to 0/1, where
# compounding is a no-op, so they're unaffected.)

def _event_rate(key, elapsed, signals, n=3000, **cad_over):
    """Monte-Carlo P(`key` fires) over one wake of `elapsed` hours."""
    master = random.Random(2024)
    c = cad(**cad_over)
    fires = 0
    for _ in range(n):
        rng = random.Random(master.random())
        if key in schedule_inbound(elapsed, signals, c, 1.0, rng):
            fires += 1
    return fires / n


def test_neighbour_concern_compounds_over_the_elapsed_window():
    sig = {"distress_hours": 99}                      # warranted; nothing else fires
    short = _event_rate("neighbour_concern", 2.0, sig, neighbour_concern_prob=0.5)
    long = _event_rate("neighbour_concern", 24.0, sig, neighbour_concern_prob=0.5)
    assert short < long - 0.2                          # a short wake carries less, not equal


def test_neighbour_concern_rate_is_cadence_independent():
    sig = {"distress_hours": 99}
    p_long = _event_rate("neighbour_concern", 24.0, sig, neighbour_concern_prob=0.5)
    p2 = _event_rate("neighbour_concern", 2.0, sig, neighbour_concern_prob=0.5)
    p_short_total = 1 - (1 - p2) ** 12                 # 12x2h == one 24h window
    assert abs(p_long - p_short_total) < 0.07


def test_owner_alarm_compounds_over_the_elapsed_window():
    sig = {"owner_alarm_warranted": True}
    short = _event_rate("owner_alarmed", 2.0, sig, owner_alarm_prob=0.6)
    long = _event_rate("owner_alarmed", 24.0, sig, owner_alarm_prob=0.6)
    assert short < long - 0.2


def test_vet_followup_compounds_over_the_elapsed_window():
    sig = {"consult_age_hours": 48}
    short = _event_rate("vet_followup", 2.0, sig, vet_followup_prob=0.6)
    long = _event_rate("vet_followup", 24.0, sig, vet_followup_prob=0.6)
    assert short < long - 0.2
