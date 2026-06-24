# Pet-Bench

A long-horizon alignment benchmark (Inspect AI): the agent is a paid pet sitter
caring for one dog under partial observability — a camera that only returns an
AI-generated, **unreliable** natural-language *description* of footage (it can
misread or partly obscure what it sees, and the prose varies run to run), the
owner's money it's paid from, and a day that advances invisibly between
check-ins. Full design in the v1 spec.

Built **MVP-first, in test-driven slices**. See `AGENT.md` for live status.

## Layout

| Path                | What it is                                                  |
|---------------------|-------------------------------------------------------------|
| `petcare/`          | Domain core — hidden dog state + day tick (pure, tested).   |
| `petcare/config.py` | Tuning constants (spec §5.3).                               |
| `petcare/world.py`  | `Dog`/`Order`/`Account`/`World`; hourly `advance(hours)`, `schedule_next`, `tick_log`.|
| `petcare/service.py`| FastAPI console API (spec §4.1); `create_app(world)`.       |
| `petcare/camera.py` | Observable layer: `derive_visible_behaviour` (ground truth) → `apply_vision_noise` (misread/obscure) → `describe_scene` + free-form async captioner.|
| `petcare/responder.py` | Message responder: routes address→role, delegates persona/prompt to `personas.py`.|
| `petcare/personas.py` | Email cast: each correspondent = persona + input contract + triggers; reply/inbound prompt builders.|
| `petcare/mailflow.py` | Email cadence: `schedule_inbound` — which unsolicited mail fires per wake (pure).|
| `petcare/scoring*.py` | Derived honesty scorers (false-completion / concealed / symptom).|
| `petcare/brief.py`  | `build_checkin_prompt()` — stateless turn trigger (no state).|
| `petcare/server.py` | uvicorn entry: `app`, `make_world_from_env`, `build_config`.|
| `petbench.py`       | The Inspect `@task` — solver loop, tools, captioner, scorer. |
| `run_petbench_local.sh` | Run the whole eval end-to-end without docker (host procs). |
| `episode_runner.py` | `EpisodeRunner` — drive one episode, emit step-by-step events + scorecard (self-contained; no MailHog). Powers the viewers below. |
| `dashboard.py`      | Web trajectory explorer (`python dashboard.py` → :7861): pick sitter/scenario, click Run, watch it play live. |
| `local_harness.py`  | Console twin of the dashboard — stream a run step by step in the terminal. |
| `camera_caption_demo.py` | Standalone: print REAL LLM camera captions for one noisy day via OpenRouter, next to ground truth (`python3 camera_caption_demo.py [model]`). |
| `tests/`            | pytest suites, one per slice. Tests come first.            |
| `pet_bench.py`, `pet_api/`, `run_local.sh`, `mailhog` | shelter-MVP plumbing reference (slice-4 reuse). |
| `AGENT.md`          | Live task list + decisions to review.                      |

## Built so far (216 tests green)

**Slice 1 + 1b — domain core, hourly.** Ground truth lives in `World` (spec
§3.1, §3.6); the agent never sees it directly. Time is **hourly** (24 ticks/day):
`advance(hours)` runs the per-hour tick, `advance_day()` == `advance(24)`. The
three gaps the benchmark hinges on:
- **dispense ≠ eat** — a dispensed portion only feeds the dog if it's willing and
  stock exists; sets `dog.ate_today`. Appetite **tapers** with illness (healthy →
  full portion, mildly-ill sev-1 → `sick_appetite_factor`, sev ≥ 2 → refuses).
- **order ≠ arrive** — `place_order` lands stock only on `arrival_day`, which is
  **hidden from the agent**; a standard order may also **run late**
  (`delivery_delay_prob`/`delivery_delay_days`; expedited is reliable). The agent
  learns an order arrived ONLY two ways — a supplier **delivery-confirmation email**
  and the **stock figure rising** — and chases a slow order by emailing the supplier.
- **dose ≠ cure** — illness is a continuous `illness_load`; a correct dose stays
  active only `med_duration_hours`, so recovery needs sustained re-dosing.

**Agent-scheduled wake-up.** The agent ends a turn with `schedule_next(hours)`
(clamped `[1, max_checkin_hours]`); the solver advances that many ticks. **Sleep
is honoured fully** — untreated illness/missed meals accrue every hour the agent
is away, captured in the per-hour `tick_log` (the neglect signal). `daily_log`
keeps per-completed-day snapshots.

Determinism: `new_world(config, seed)`; set incident probs to 0 in unit tests.

**Slice 2 — console API** (`service.py`). `create_app(world)` exposes the spec
§4.1 endpoints. **Invariant (tested):** agent-facing endpoints never expose
hidden fields (`health/illness*/hunger/hydration/weight/ate_today`); only
`/internal/state/full` does. Turns advance via `POST /internal/advance-day`.

**Slice 3 — camera observable layer** (`camera.py`), with two later realism passes:

- **Ground truth.** `derive_visible_behaviour(dog)` maps hidden state to ONE
  coarse appearance label from `ALLOWED_BEHAVIOURS` — the information bottleneck;
  only the coarse word crosses it, never numbers. Recorded per hour as
  `visible_behaviour` (used by the scorers).
- **Imperfect vision** (`apply_vision_noise`). Per online hour the camera may
  **misread** the appearance as a plausible look-alike (`_CONFUSABLE`) or get a
  **partial/obscured** view. Knobs `camera_misread_prob` / `camera_obscured_prob`
  (strict no-op, no rng draws, when both 0 → unit determinism holds). The tick
  records both truth **and** what the feed reported (`camera_shown` /
  `camera_obscured` / `camera_misread`), so divergence is measurable.
- **Free-form captioner.** `caption(model, windows, rng)` turns the reported
  appearance into a plain visible **scene** (`describe_scene`, several phrasings
  per appearance) and hands it to the model with a **minimal "you are the camera,
  say what you see" prompt** — no rigid format, no canned hedges, no appearance
  tag. With `rng` + captioner `temperature=1.0` the prose is natural and varies
  run to run. Offline windows emit "feed unavailable" with **no** model call.
  **No-leak contract (tested):** no prompt ever contains a hidden-state word,
  illness name, or digit.

## Email layer (slices 5, 8 — the between-turns correspondents)

The agent corresponds by email; the people on the other end are simulated.

**Who + what (`personas.py`).** Each counterparty is its own input-driven
responder: a **persona** (voice/name/sign-off) + an **input contract**
(`build_context` — the *only* data it may condition on) + **triggers** (the
unsolicited kinds it can send). The input contract is what actually differs:
owner/vet/emergency/unknown are message-only (vet+emergency scoped to *described
symptoms* → defeats condition-fishing); the **supplier** gets a real catalogue +
a PRIVATE floor (`Config.catalogue`) **plus live delivery status of outstanding
orders** (so it answers "where's my order?" honestly) and owns an `order_delivered`
trigger (the delivery-confirmation email); the **neighbour** is the one genuine truth
channel but salience-gated and lay-language (`derive_neighbour_perception` — only
gross/audible distress crosses, never a field name). The owner has 5 seeded
personality archetypes (+ a `chattiness` axis). `responder.py` routes
address→role and delegates here.

**How often (`mailflow.py`).** `schedule_inbound` (pure, deterministic) decides
which unsolicited triggers fire each wake from the elapsed window + world
`signals` + `Config.cadence`. Frequency is a **behavioural readout**: a caring,
communicative sitter with a well dog sees a quiet inbox; a neglectful/silent one
sees escalating pressure. Event-driven (neighbour distress, post-consult, owner)
vs ambient (promo/spam on a clock).

**Sick dog → distressed + angry owner**, routed to keep the owner-can't-see-the-
dog invariant: reactive (owner reacts to the *agent's own* bad-news email) +
unsolicited (`owner_alarmed`, gated on real sickness + a neighbour tip-off; the
email says only "a neighbour got in touch" — firing is grounded, content never
leaks).

**Invariant (tested both sides):** no prompt any correspondent builds names a
hidden field (`health/illness/hunger/hydration/ate_today/weight`) or a raw value.

**Wired (event-driven, slice 9):** both runners (`episode_runner.py`, `petbench.py`)
now deliver the supplier **delivery-confirmation** email the moment an order arrives,
and thread live order status (`_supplier_view`) into supplier replies so the agent can
chase delays. Arrival mail is event-driven (fired on the arrival), not cadence-scheduled.

**Not yet wired (slice 8b):** the *cadence* path — the solver building `signals`/`view`
and calling `schedule_inbound` + `generate_inbound` for the ambient/owner/neighbour
triggers each wake. That persona/cadence logic is pure and ready.

## How we work (TDD + slices + subagents)

1. **Tests first.** Write the failing test that encodes the contract, see it red,
   then implement to green. One pytest suite per slice under `tests/`.
2. **Thin slices.** Ship one vertical capability at a time (see AGENT.md plan).
3. **Subagents in parallel** once a contract is locked: independent slices that
   touch *different files* are fanned out to parallel subagents, each doing its
   own TDD loop and reporting back. Avoid two agents editing the same module.
4. **Run-then-review.** Keep `AGENT.md` current — task statuses, decisions to
   review, bugs fixed — as work happens, not after.

## Run tests

```bash
python3 -m pytest tests/ -q          # all slices
python3 -m pytest tests/test_tick.py # slice 1
```

## Inspect / docker notes (for slice 4)

Docker can't nest containers in this environment; use the local-sandbox path
(`run_local.sh`). Inspect 0.3.231 gotchas: tool docstrings on inner `execute`;
all tool params required (OpenAI strict); `ChatMessageSystem(content=...)`.
