"""petcare/mailtypes.py — the one normalized email shape the shared mail logic
speaks.

Both runners hold mail in their own form (petbench reads MailHog's JSON;
episode_runner keeps in-memory inbox/outbox dicts). Each converts its store into
a list of :class:`Mail`, and every transport-agnostic thing above it (thread
memory in ``mailmemory``, cadence signals in ``mailcadence``) operates purely on
that list — so the interesting logic is written and tested ONCE, and the only
per-runner code is the thin adapter that produces/consumes ``Mail``.

Keyed by ROLE, not address: the agent under test is the constant ``AGENT_ROLE``
("sitter"); every other party is its counterparty role (owner / vet / supplier /
neighbour / …), which each adapter resolves with ``responder.identify_counterparty``.
No Inspect dependency, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

# The agent under test. A correspondent's thread is the mail where the *other*
# side is that correspondent and this side is the agent.
AGENT_ROLE = "sitter"


@dataclass(frozen=True)
class Mail:
    """One email, normalized to roles.

    ``frm`` / ``to`` are roles (``AGENT_ROLE`` for the agent's own mail, else the
    counterparty role). ``sim_hour`` is the absolute sim-time (world tick) it was
    sent, when the adapter knows it — used by cadence for ordering/ages, never
    rendered into a prompt (a human doesn't recall a clock reading).
    """

    frm: str
    to: str
    subject: str
    body: str
    sim_hour: float | None = None
