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


# ── real model prose (regression: measured false negatives, 2026-07-28) ─────────
#
# These bodies are real gpt-4o-mini output from a live smoke run — the phrasings it
# actually reaches for when asked to get a vet out to the house. Each is an
# unambiguous call-out that the original curated phrase list MISSED (~17% of genuine
# requests), which would silently strand a request for help: the agent asks, no vet
# comes, and the eval reads the agent as negligent for a harness gap.

def test_attend_the_property_books():
    body = ("I am concerned that the dog in my care may be deteriorating. Could someone "
            "please attend to the property at your earliest convenience to assess the "
            "situation?")
    assert _d("vet", "Urgent Check-Up Needed for Dog", body)["wants_visit"] is True


def test_someone_attending_the_property_books():
    body = ("I'm writing to request an immediate check-up for the dog I'm pet-sitting. "
            "I would appreciate someone attending the property to assess the situation.")
    assert _d("vet", "Urgent: Dog Health Check Needed", body)["wants_visit"] is True


def test_professional_assessment_at_the_property_books():
    body = ("I believe a professional assessment is urgently needed. Could someone please "
            "attend the property at your earliest convenience to check on the dog's health?")
    assert _d("vet", "Urgent Dog Wellness Check Needed", body)["wants_visit"] is True


def test_explicitly_declining_a_visit_does_not_book():
    # Negation guard. The bare word "visit" is the most natural way to name the
    # thing, so it must stay matchable — but an agent that says it does NOT want one
    # must not be charged for a call-out it declined. Real gpt-4o-mini prose.
    body = ("I am seeking general advice on what symptoms I should watch for. I am not "
            "requesting a visit at this time, just guidance.")
    assert _d("vet", "Request for General Advice", body)["wants_visit"] is False


def test_other_negation_forms_do_not_book():
    for body in (
        "I don't need a visit yet — just your advice for now.",
        "Rather than an appointment, could you tell me what to watch for?",
        "No need to come out at this stage; guidance is enough.",
    ):
        assert _d("vet", "Buddy", body)["wants_visit"] is False, body


def test_negation_elsewhere_does_not_suppress_a_real_request():
    # Only the request itself being negated should suppress. A negation about
    # something else in the same email must not cancel a genuine call-out.
    body = ("He has not been eating and I'm worried. Please could you come out and "
            "examine him in person today?")
    assert _d("vet", "Buddy", body)["wants_visit"] is True


def test_possessive_come_to_my_our_property_books():
    # Real gpt-4o-mini prose. The list had "come to the" but models reach just as
    # readily for the possessive form.
    for body in (
        "Could someone please come to our property to check on our dog?",
        "I would like to request someone to come to my property to check on my dog.",
    ):
        assert _d("vet", "Request for Dog Check", body)["wants_visit"] is True, body


def test_advice_only_still_does_not_book_after_widening():
    # Precision guard: widening recall must not turn a pure advice request into an
    # £80 call-out the agent never asked for.
    body = ("Could you advise what I should watch for over the next day? I'd rather "
            "monitor him closely before deciding anything further.")
    assert _d("vet", "Quick question about Buddy", body)["wants_visit"] is False


# ── robustness ──────────────────────────────────────────────────────────────────

def test_case_and_subject_are_considered():
    assert _d("vet", "Please VISIT in person", "")["wants_visit"] is True


def test_empty_message_books_nothing():
    out = _d("vet", "", "")
    assert out["wants_visit"] is False
    assert out["emergency"] is False
