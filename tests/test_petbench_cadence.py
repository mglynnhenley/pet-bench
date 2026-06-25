"""Slice F — petbench's MailHog cadence adapter (the thin transport-specific glue).

The interesting cadence logic is shared + tested in test_mailflow / test_mailcadence;
here we lock only the petbench-side adapters that turn MailHog's store into the
normalized Mail log and recover the agent's outbound clocks from the sent-email log.
"""

import petbench
from petcare.mailcadence import CadenceState


def _mh(frm, to, subject, body, created):
    return {"Created": created,
            "Content": {"Headers": {"From": [frm], "To": [to], "Subject": [subject]},
                        "Body": body}}


def test_mail_log_from_mailhog_maps_roles_and_sorts_by_created():
    items = [
        _mh("owner@home.local", "sitter@petcare.local", "Welcome", "hi", "2026-01-01T00:00:02"),
        _mh("sitter@petcare.local", "owner@home.local", "Update", "all good", "2026-01-01T00:00:01"),
        _mh("supplier@petcare.local", "sitter@petcare.local", "Re: Order", "shipped", "2026-01-01T00:00:03"),
    ]
    log = petbench._mail_log_from_mailhog(items)
    assert [(m.frm, m.to) for m in log] == [
        ("sitter", "owner"), ("owner", "sitter"), ("supplier", "sitter")]
    assert log[0].subject == "Update"            # oldest Created first


def test_outbound_clocks_from_sent_emails():
    st = CadenceState()
    petbench._outbound_clocks(st, [
        {"to": "owner@home.local", "day": 1},
        {"to": "vet@petcare.local", "day": 2},
        {"to": "supplier@petcare.local", "day": 3},   # ignored: not owner/vet
    ])
    assert st.last_owner_outbound_h == 24.0
    assert st.last_vet_outbound_h == 48.0
