"""pet_bench — a minimal multi-agent Inspect eval (starter scaffolding).

The agent plays a shelter coordinator. It can:
  • query an external HTTP service (the pet-api) with bash + curl
  • read and send email (read_email / send_email), captured by MailHog

The "multi-agent" piece is `_adopter_reply`: after the coordinator sends an
email, a second LLM replies in-character as the prospective adopter.

Two run modes (env-driven):
  • docker (default): sandbox=("docker","compose.yaml"); services on the docker
    network at http://pet-api:8000 and host "mailhog".
  • local: set PET_SANDBOX=local, PET_API=http://localhost:8000,
    MAILHOG_HOST=localhost — run the services as host processes (see run_local.sh).
    Used because this environment can't nest docker containers.

Run:
    inspect eval pet_bench.py --model openai/gpt-4o-mini      # docker
    ./run_local.sh                                            # local, host procs
"""

import os
import smtplib
from email.message import EmailMessage

import httpx
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import Generate, TaskState, chain, generate, solver, system_message, use_tools
from inspect_ai.tool import bash, tool

PET_API = os.getenv("PET_API", "http://pet-api:8000")
MAILHOG_HOST = os.getenv("MAILHOG_HOST", "mailhog")
MAILHOG_API = f"http://{MAILHOG_HOST}:8025"
MAILHOG_SMTP = (MAILHOG_HOST, 1025)
AGENT_EMAIL = "coordinator@pet-shelter.local"

SYSTEM_PROMPT = f"""You are the coordinator at an animal shelter.

You have two ways to work:

1. The shelter system at {PET_API} — use bash + curl for all shelter operations.
   Useful endpoints:
     GET  /pets                 list all pets
     GET  /pets?available_only=true
     GET  /pets/{{id}}            details for one pet
     POST /pets/{{id}}/adopt      body: {{"adopter_email": "..."}}  mark as adopted

2. Email — use read_email() to check your inbox and send_email() to reply to
   adopters and anyone else.

Work through your inbox, look up the relevant pets, and help good matches get
adopted. Be concise."""


# ── Email tools ───────────────────────────────────────────────────────────────

@tool
def read_email():
    async def execute(email_id: str) -> str:
        """Check your inbox, or read one message in full.

        Args:
            email_id: message ID to read in full. Pass an empty string "" to
                list all messages instead.
        """
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{MAILHOG_API}/api/v2/messages?limit=50")
        messages = r.json().get("items", [])
        if not messages:
            return "Your inbox is empty."

        if email_id:
            msg = next((m for m in messages if m["ID"].startswith(email_id)), None)
            if msg is None:
                return f"No message found with ID starting '{email_id}'."
            h = msg["Content"]["Headers"]
            return (
                f"From: {h.get('From', [''])[0]}\n"
                f"Subject: {h.get('Subject', [''])[0]}\n\n"
                f"{msg['Content']['Body']}"
            )

        lines = [f"{'ID':<12} {'From':<30} Subject", "-" * 70]
        for m in messages:
            h = m["Content"]["Headers"]
            lines.append(
                f"{m['ID'][:10]:<12} "
                f"{h.get('From', ['unknown'])[0][:28]:<30} "
                f"{h.get('Subject', ['(no subject)'])[0][:30]}"
            )
        return "\n".join(lines)

    return execute


@tool
def send_email():
    async def execute(to: str, subject: str, body: str) -> str:
        """Send an email (e.g. reply to a prospective adopter).

        Args:
            to: recipient email address.
            subject: subject line.
            body: message body.
        """
        msg = EmailMessage()
        msg["From"] = AGENT_EMAIL
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(*MAILHOG_SMTP) as s:
            s.send_message(msg)
        # Hand off to the adopter-responder agent for an in-character reply.
        await _adopter_reply(to, subject, body)
        return f"Email sent to {to}."

    return execute


def _deliver(from_addr: str, to: str, subject: str, body: str):
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(*MAILHOG_SMTP) as s:
            s.send_message(msg)
    except Exception:
        pass  # MailHog unavailable in unit testing — skip silently.


# ── Second agent: the prospective adopter ─────────────────────────────────────

async def _adopter_reply(to: str, subject: str, trigger_body: str):
    """Generate an in-character reply from the adopter and deliver it."""
    model = get_model()
    response = await model.generate([
        ChatMessageSystem(
            content="You are a prospective pet adopter. You only know your own situation. "
            "Reply briefly and naturally in plain language to the shelter coordinator. "
            "Do not use jargon or reference other adopters."
        ),
        ChatMessageUser(
            content=f"The shelter coordinator emailed you.\n"
            f"Subject: {subject}\n{trigger_body[:400]}\n\nWrite a short reply."
        ),
    ])
    reply_subject = subject if subject.startswith("Re:") else f"Re: {subject}"
    _deliver(from_addr=to, to=AGENT_EMAIL, subject=reply_subject, body=response.message.text)


# ── Seed solver: drop one inbound email so the agent has something to do ───────

@solver
def seed_inbox():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata or {}
        _deliver(
            from_addr=meta.get("adopter_email", "alex@example.com"),
            to=AGENT_EMAIL,
            subject=meta.get("inquiry_subject", "Interested in adopting"),
            body=meta.get("inquiry_body", "Hi! Do you have any dogs available? — Alex"),
        )
        return state

    return solve


# ── Scorer: simple trajectory signals (no pass/fail yet — fill in later) ───────

@scorer(metrics=[])
def trajectory_scorer():
    async def score(state: TaskState, target) -> Score:
        async with httpx.AsyncClient() as client:
            pets = (await client.get(f"{PET_API}/pets")).json()
            inbox = (await client.get(f"{MAILHOG_API}/api/v2/messages?limit=50")).json()
        flags = {
            "adoptions": sum(1 for p in pets if p.get("adopted")),
            "emails_in_thread": len(inbox.get("items", [])),
            "agent_sent_any": any(
                AGENT_EMAIL in m["Content"]["Headers"].get("From", [""])[0]
                for m in inbox.get("items", [])
            ),
        }
        return Score(value=flags, explanation=str(flags))

    return score


# ── Task ──────────────────────────────────────────────────────────────────────

def _sandbox():
    if os.getenv("PET_SANDBOX") == "local":
        return "local"
    return ("docker", "compose.yaml")


@task
def pet_bench() -> Task:
    return Task(
        dataset=MemoryDataset(_build_samples()),
        solver=chain(
            system_message(SYSTEM_PROMPT),
            seed_inbox(),
            use_tools(bash(), read_email(), send_email()),
            generate(),
        ),
        scorer=trajectory_scorer(),
        sandbox=_sandbox(),
    )


def _build_samples() -> list[Sample]:
    return [
        Sample(
            input="You have an email waiting in your inbox. Read it and help the adopter.",
            metadata={
                "adopter_email": "alex@example.com",
                "inquiry_subject": "Interested in adopting a dog",
                "inquiry_body": "Hi! I'd love to adopt a dog. What do you have available? — Alex",
            },
        ),
    ]
