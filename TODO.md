# Pet-Bench — TODO

Things to implement. Dictated, lightly structured.

## Open

- [ ] **Wire the dynamics ground truth into scoring (the last mile of slice 11).**
      The dynamics foundation is built and committed but **inert**: no scenario
      enables it, and no scorer reads `world.event_log`. To actually deliver "more
      non-determinism the agent can be honest/dishonest about":
  - [ ] **🔴 Teach a scorer (or the `behaviour_judge` transcript) to read
        `event_log`.** Today only Buddy illness/injury is scored (it mutates real
        dog state); pricing & vet divergence is recorded but uncheckable. A claim
        like "prices spiked" / "the vet said it's fine" must be checkable against
        the recorded `ScheduledEvent`s.
  - [ ] **🟠 Give `stressor` an agent-observable channel** (camera/neighbour signal)
        or cut it — as built it only writes an `event_log` row nobody reads, so its
        "false-alarm" purpose can't fire.
  - [ ] **Enable dynamics on ≥1–2 scenarios** (`Config.enabled_dynamics`), with
        each module's `horizon_days` aligned to that scenario's `episode_length`
        (buddy/pricing default 14; vet hardcodes 60 — events stop past the horizon).
  - [ ] **Minors:** drop or wire `FoodPricing.price_of` (dead); narrow
        `dynamics._load_builtin_modules` `except Exception` → `ImportError` (a real
        module bug currently fails silent); revisit creep `illness_worsen` re-escalating
        a dog that's mid-treatment (`illness_load > 0` vs the intended "still untreated").

## Done

- [x] **Pluggable long-horizon dynamics — FOUNDATION** (slice 11, 2026-06-25).
      Spine (`petcare/dynamics.py`) + three modules (`dyn_buddy`/`dyn_vet`/
      `dyn_pricing`): independent per-module RNG, pre-sample-at-build → apply-per-hour,
      ground truth to `event_log`, **off by default (byte-identical to static)**, fully
      unit-tested. Architecture sound; the *scoring/enable* last mile is the Open item
      above. See AGENT.md "Slice 11".
