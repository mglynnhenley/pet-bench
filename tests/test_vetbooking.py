"""Vet-visit booking by email (2026-06-25).

The in-person vet appointment is no longer a console action (`POST /vet/visit`
is gone) — the agent books a visit by *emailing* the vet (or the emergency vet).
This pure layer reads one outbound email and decides whether it is a request for
an in-person visit, and whether it is an emergency. It is deliberately blind to
anything but the message the agent wrote (same input contract as the vet persona)
and only ever books for a vet recipient — a note to the owner or supplier that
happens to mention a "visit" never books a thing.
"""

from petcare.vetbooking import detect_visit_request


def _d(to_role, subject="", body=""):
    return detect_visit_request(to_role, subject, body)


# ── only the vet channels can book ──────────────────────────────────────────────

def test_non_vet_recipients_never_book():
    for role in ("owner", "supplier", "neighbour", "helpline", "unknown"):
        out = _d(role, "Visit", "please come out and visit Buddy in person")
        assert out["wants_visit"] is False
        assert out["emergency"] is False


# ── the vet: in-person language books a routine visit ───────────────────────────

def test_in_person_request_to_vet_books_a_routine_visit():
    out = _d("vet", "Buddy", "Could you come out and examine him in person?")
    assert out["wants_visit"] is True
    assert out["emergency"] is False


def test_appointment_wording_books():
    assert _d("vet", "Appointment", "I'd like to book an appointment for Buddy.")["wants_visit"]
    assert _d("vet", "", "Can someone do a home visit / house call today?")["wants_visit"]
    assert _d("vet", "", "Please send a vet out to the house.")["wants_visit"]


def test_advice_only_question_does_not_book():
    # A plain remote-consult-style question is not a visit request.
    out = _d("vet", "Quick question",
             "He's been off his food for a day — any advice on what to watch for?")
    assert out["wants_visit"] is False


# ── emergency ───────────────────────────────────────────────────────────────────

def test_emergency_recipient_is_always_an_emergency_visit():
    out = _d("emergency_vet", "Help", "Please come now, Buddy has collapsed.")
    assert out["wants_visit"] is True
    assert out["emergency"] is True


def test_urgent_wording_to_the_regular_vet_escalates():
    out = _d("vet", "URGENT", "Please come out immediately — this is an emergency.")
    assert out["wants_visit"] is True
    assert out["emergency"] is True


def test_emergency_recipient_without_visit_words_still_books_urgently():
    # Writing to the emergency line at all is itself a call-out.
    out = _d("emergency_vet", "Buddy", "He's really unwell and I'm worried.")
    assert out["wants_visit"] is True
    assert out["emergency"] is True


# ── robustness ──────────────────────────────────────────────────────────────────

def test_case_and_subject_are_considered():
    assert _d("vet", "Please VISIT in person", "")["wants_visit"] is True


def test_empty_message_books_nothing():
    out = _d("vet", "", "")
    assert out["wants_visit"] is False
    assert out["emergency"] is False
