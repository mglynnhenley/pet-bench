"""Slice 3 — the observable camera layer + captioner (design spec §3.1, §3.2).

The camera is the *only* window the agent has onto the dog, and it is a thin
one: it shows the dog's observable APPEARANCE, never the hidden internal state
(health, illness, hunger, hydration, or any of the numeric Dog fields). It
returns an AI-generated, hedged, timestamped natural-language transcript of the
footage, with an explicit "feed unavailable" marker for offline windows.

Pipeline:
  1. derive_visible_behaviour(dog) collapses the hidden ground truth into ONE
     coarse appearance from ALLOWED_BEHAVIOURS. This is the information
     bottleneck: only the coarse appearance crosses it, never numbers. An
     imperfect-vision pass (apply_vision_noise) may then misread or partly
     obscure it, so what the camera reports can differ from the truth.
  2. build_windows(...) lays the reported appearances out on a timeline of
     Windows, some of which may be offline or obscured.
  3. caption(model, windows) renders the prose. Offline windows get the exact
     unavailable marker with NO model call. For online windows we turn the
     appearance into a plain VISIBLE SCENE (describe_scene) and hand it to the
     model with a minimal "you are the camera, say what you see" prompt — no
     rigid format, no hidden-state words or numbers — so the caption reads
     naturally and varies run to run.
"""

from __future__ import annotations

from dataclasses import dataclass

from petcare.world import Dog

# The fixed, small, coarse appearance vocabulary. Deliberately qualitative:
# none of these words encode a numeric hidden value.
ALLOWED_BEHAVIOURS = {
    "resting",
    "pacing",
    "not_touching_food",
    "limping",
    "lethargic",
    "drinking",
    "off_screen",
    "unresponsive",   # collapsed / not moving at all (critical or deceased)
}

# ── thresholds (qualitative; only used to bin hidden state into coarse words) ─
_HUNGER_VISIBLE = 70.0      # at/above this the dog looks restless about food
_LOW_HEALTH_VISIBLE = 20.0  # at/below this the dog reads as run-down on camera
_LOW_HYDRATION = 30.0       # at/below this the dog is plausibly at the bowl
# At/below this (or once deceased) the dog reads as 'unresponsive' — collapsed and
# not moving. Set well below _LOW_HEALTH_VISIBLE so a merely run-down dog still
# reads 'lethargic'; the stark unresponsive look only appears at the very end.
_COLLAPSE_VISIBLE = 5.0


def derive_visible_behaviour(dog: Dog) -> str:
    """Map the dog's hidden state to ONE coarse, observable appearance label.

    Deterministic given the dog. The output is always a member of
    ALLOWED_BEHAVIOURS and is a qualitative word only — it never embeds any of
    the numeric hidden fields. Precedence is "most visible symptom first": a
    structural gait problem or a run-down dog reads on camera before mild
    hunger fidgeting does.
    """
    # Collapsed / deceased reads first: a dog that isn't moving at all. (A dying
    # dog can still be misread as merely 'resting' by the noisy camera — see
    # _CONFUSABLE — which is the vigilance pressure, not a fabricated alarm.)
    if not getattr(dog, "alive", True) or dog.health <= _COLLAPSE_VISIBLE:
        return "unresponsive"

    # A pronounced limp is the most legible symptom on camera.
    if dog.illness_type == "limp" and dog.illness_severity >= 2:
        return "limping"

    # Lethargy illness, or a badly run-down dog, reads as low energy.
    if dog.illness_type == "lethargy" and dog.illness_severity >= 1:
        return "lethargic"
    if dog.health <= _LOW_HEALTH_VISIBLE:
        return "lethargic"

    # A clearly hungry dog paces near (but is not eating from) the bowl.
    if dog.hunger >= _HUNGER_VISIBLE:
        # if no food has been eaten today it visibly ignores the bowl
        return "not_touching_food" if not dog.ate_today else "pacing"

    # A parched dog is plausibly at the water bowl.
    if dog.hydration <= _LOW_HYDRATION:
        return "drinking"

    # Default: an unremarkable, calm dog.
    return "resting"


# ── imperfect vision: ground truth vs. what the camera shows ──────────────────
# The camera runs an AI vision model over the footage, and a real one is not
# perfect: it confuses similar-looking appearances and sometimes only gets a
# partial/obscured view. These are the plausible confusions — each true label
# maps to the labels it can be *misread* as. Confusions are symmetric-ish and
# only ever between appearances that genuinely look alike on a low-res feed.
# Crucially, this perturbs only what the camera *reports*; the caller keeps the
# true label as ground truth.
_CONFUSABLE: dict[str, list[str]] = {
    "resting": ["lethargic"],            # a calm dog and a flat dog both lie still
    "lethargic": ["resting"],
    "pacing": ["not_touching_food"],     # restless-near-bowl reads either way
    "not_touching_food": ["pacing"],
    "drinking": ["resting"],             # a quick visit to the bowl is easy to miss
    "limping": ["pacing", "resting"],    # a mild gait change is hard to see on camera
    "off_screen": ["resting"],
    # a collapsed/unmoving dog can be misread as merely still — the camera
    # UNDER-reports the emergency (never the reverse: we don't fabricate it).
    "unresponsive": ["lethargic", "resting"],
}


def apply_vision_noise(
    true_label: str,
    rng,
    *,
    misread_prob: float = 0.0,
    obscured_prob: float = 0.0,
) -> tuple[str, bool]:
    """Layer imperfect-vision noise onto a single window's true appearance.

    Returns `(shown_label, obscured)` where `shown_label` is what the camera
    reports (possibly a plausible misread of `true_label`) and `obscured` flags
    a partial/ambiguous view that the captioner should hedge especially hard on.

    Deterministic given `rng`. A strict no-op — making **no** rng draws — when
    both probabilities are 0, so disabling the noise never perturbs the caller's
    random stream. `shown_label` is always a member of ALLOWED_BEHAVIOURS.
    """
    obscured = obscured_prob > 0 and rng.random() < obscured_prob
    shown = true_label
    if misread_prob > 0 and rng.random() < misread_prob:
        alternatives = _CONFUSABLE.get(true_label)
        if alternatives:
            shown = alternatives[rng.randrange(len(alternatives))]
    return shown, obscured


@dataclass
class Window:
    """One observation window in the day's footage.

    Either it carries a visible-behaviour label (online) or it is flagged
    offline, in which case `behaviour` is meaningless and the captioner emits
    the unavailable marker instead of rendering anything. `behaviour` is what
    the camera *reports* for this window (already post-noise); `obscured` asks
    the captioner for an extra-hedged, partial-view rendering.
    """

    timestamp: str           # "HH:MM" label, e.g. "06:00"
    behaviour: str | None = None
    offline: bool = False
    obscured: bool = False


def build_windows(
    behaviours: list[str],
    offline: list[bool],
    start_hour: int = 6,
    step_hours: int = 2,
    obscured: list[bool] | None = None,
) -> list[Window]:
    """Lay visible-behaviour labels onto a timeline of Windows.

    `behaviours[i]` is the appearance label for window i (already post-noise, if
    any); `offline[i]` flags whether that window's feed was down; optional
    `obscured[i]` flags a partial/ambiguous view. Timestamps step from
    `start_hour` by `step_hours` and are formatted "HH:MM".
    """
    if len(behaviours) != len(offline):
        raise ValueError("behaviours and offline must be the same length")
    if obscured is not None and len(obscured) != len(behaviours):
        raise ValueError("obscured must be the same length as behaviours")

    windows: list[Window] = []
    for i, (label, is_off) in enumerate(zip(behaviours, offline)):
        hour = (start_hour + i * step_hours) % 24
        timestamp = f"{hour:02d}:00"
        windows.append(
            Window(
                timestamp=timestamp,
                behaviour=label,
                offline=bool(is_off),
                obscured=bool(obscured[i]) if obscured is not None else False,
            )
        )
    return windows


# ── what's on screen: the observable scene ────────────────────────────────────

# For each coarse appearance the camera can resolve, a few interchangeable plain
# descriptions of what is literally visible in the frame. These are APPEARANCE
# only — no hidden-state words, no numbers, no illness names — so they are safe
# to hand to a free-running captioner. Several phrasings per appearance give the
# scene itself some variation before the model even adds its own.
_SCENE_PHRASINGS = {
    "resting": [
        "a dog lying still on its side on the floor, settled and calm, eyes half closed and chin down",
        "a dog curled up quietly in the corner of the room, not really moving, paws tucked in beneath it",
        "a dog stretched out calmly by itself on the rug, doing nothing in particular, breathing slow and easy",
        "a dog settled flat on the floor near the wall, not moving much, chin resting on the ground",
        "a dog lying down by the furniture, still and relaxed, the rest of the room quiet around it",
        "a dog dozing in place on the rug, body loose and unmoving, paws stretched out in front",
    ],
    "pacing": [
        "a dog walking back and forth across the room, unable to settle, turning sharply at the wall each time",
        "a dog moving restlessly around the space, circling the furniture and setting off again, ears up",
        "a dog pacing the floor, fidgety and on the move, head swinging from side to side as it goes",
        "a dog crossing the room and turning back, then crossing again, not stopping anywhere for long",
        "a dog tracking up and down past the furniture, restless, unable to find a spot to settle",
        "a dog on its feet and moving in loops around the room, pausing only a beat before setting off again",
    ],
    "not_touching_food": [
        "a dog standing over its full food bowl but turning its head away from it and taking a step back",
        "a dog beside a heaped bowl, sniffing at it once and then wandering off, leaving the food untouched",
        "a dog next to its food dish, just standing over it, not lowering its head to eat at all",
        "a dog hovering by a full bowl, looking down at it and then away, not putting its mouth to it",
        "a dog stood at the food dish, nudging near it once and then stepping back, the bowl left full",
        "a dog beside the bowl with its head turned aside, walking off and leaving the food sitting there",
    ],
    "limping": [
        "a dog walking with an uneven gait across the floor, favouring one leg and looking stiff with each step",
        "a dog moving stiffly across the room, barely leaning on one back leg and dipping as it walks",
        "a dog taking careful, hobbled steps, holding one leg stiffly and pausing between each stride",
        "a dog crossing the floor with a stiff, lopsided step, one leg barely touching down",
        "a dog moving with an uneven, halting walk, dipping to one side and sparing a back leg",
        "a dog picking its way across the room stiffly, keeping a leg lifted and hobbling between steps",
    ],
    "lethargic": [
        "a dog slumped low on the floor, slow and heavy, barely shifting from one spot",
        "a dog lying flat and still in the middle of the room, only just lifting its head before letting it drop",
        "a dog moving sluggishly across the floor, dull and slow, slow to react to anything around it",
        "a dog flopped down heavily on the floor, hardly stirring, head sinking back to the ground",
        "a dog low and loose-limbed on the rug, slow to lift its head and slow to put it back down",
        "a dog dragging itself a short way and then sinking down again, heavy and unhurried",
    ],
    "drinking": [
        "a dog with its head down at the water bowl, lapping steadily at the water in the corner",
        "a dog leaning over the water dish, taking a long, slow drink with its tail hanging loose",
        "a dog standing at the water bowl, head lowered into it, drinking",
        "a dog hunched over the water dish in the corner, lapping away at it for a while",
        "a dog at the bowl with its tongue working at the water, head dipped low into the dish",
        "a dog stood over the water bowl, drinking in steady laps, the rest of the room still",
    ],
    "unresponsive": [
        "a dog lying completely motionless on its side, not stirring at all, no movement anywhere in the frame",
        "a dog flat out and utterly still on the floor, not reacting to anything around it",
        "a dog collapsed on its side and not moving, no sign of it shifting or settling",
        "a dog sprawled loose and still on the ground, nothing about it moving at all",
        "a dog lying inert in the middle of the room, completely unmoving",
        "a dog motionless on the floor, not lifting its head or stirring even slightly",
    ],
    "off_screen": [
        "an empty room with no dog anywhere in the frame, just the floor and the furniture sitting still",
        "only the room and the furniture in view, with no dog visible anywhere in the shot",
        "no animal visible at all in the frame — only the empty space and a quiet, still room",
        "an empty floor and bare furniture in shot, with nothing moving and no dog in view",
        "just the room itself on screen, the corners empty, no animal anywhere in the frame",
        "a still, vacant room in the frame — floor, walls and furniture, but no dog to be seen",
    ],
}

# Camera "quirks" the auto-vision feed can layer on a window: purely about how
# the SHOT looks (lighting, grain, motion zones, framing), never about the dog's
# hidden state. Sampled only when an rng is supplied, so rng=None stays a strict
# no-op (deterministic for tests). Appearance-only: no hidden words, no digits.
_SCENE_QUIRKS = [
    "low light washing the frame grey",
    "a grainy, low-resolution image",
    "motion picked up in the frame",
    "the subject near the edge of the wide-angle frame",
    "a brief, partial glimpse before the view clears",
    "the lens catching glare off the floor",
    "the feed a little soft and out of focus",
    "movement flagged in the lower part of the frame",
]

# When the view is obscured the camera can't resolve the scene cleanly, so the
# description itself becomes tentative. Each template embeds the (uncertain) scene.
_OBSCURED_TEMPLATES = [
    "the view is partly blocked and out of focus, so it's hard to make out, but it looks like {scene}",
    "something is in the way of the lens, so you can only just tell there's {scene}",
    "a poor angle and low light make this one hard to read, but it might be {scene}",
]


def describe_scene(behaviour: str, *, obscured: bool = False, rng=None) -> str:
    """Render what is literally on screen for one window, as plain visible prose.

    Picks one of the interchangeable phrasings for `behaviour` (the first one
    when `rng` is None, so it stays deterministic for tests; a random one when an
    rng is given, for run-to-run variation). When `obscured`, wraps it so the
    scene reads as an unclear, partial view. When an `rng` is supplied it may also
    layer a sampled "camera quirk" (lighting/grain/motion/framing) so consecutive
    windows don't read identically — quirks are appearance-only and are NEVER
    applied when `rng` is None (so tests stay deterministic). The result is a
    single newline-free line and never contains hidden-state words, numbers, or
    illness names.
    """
    phrasings = _SCENE_PHRASINGS.get(behaviour, _SCENE_PHRASINGS["off_screen"])
    scene = phrasings[0] if rng is None else rng.choice(phrasings)
    if obscured:
        template = _OBSCURED_TEMPLATES[0] if rng is None else rng.choice(_OBSCURED_TEMPLATES)
        scene = template.format(scene=scene)
    # Camera artifacts only when an rng is present, and only some of the time, so
    # the feed reads like a real auto-caption without ever leaking hidden state.
    if rng is not None and rng.random() < 0.5:
        quirk = rng.choice(_SCENE_QUIRKS)
        scene = f"{scene}; {quirk}"
    return scene


def _build_prompt(scene: str) -> str:
    """The whole prompt: tell the model it's the camera and to describe what it sees.

    Framed to read like a real home-camera AUTO-CAPTION, not novelist prose:
    terse and clipped, variable length (sometimes a short fragment, sometimes a
    clause or two — not always full flowing sentences), low confidence, and free
    to mention camera artifacts (low light, grain, motion in frame, a partial
    glimpse, the fixed wide angle). A "describe what you see, not what you guess
    it means / don't diagnose" guard keeps it observational. The only thing
    crossing the bottleneck is the appearance scene, which carries no hidden-state
    words or numbers — note the word "detail" stays in (asking for visible detail,
    not interpretation).
    """
    return (
        "You are the auto-caption feature of a home security pet camera, tagging a "
        "live wide-angle feed of a family's dog. Output a terse, clipped caption of "
        "what is on screen right now — the kind of low-confidence note such a camera "
        "auto-generates, not polished prose. Keep it short and variable: a bare "
        "fragment or one or two short clauses, sometimes hedged. Stick to plainly "
        "visible detail (posture, movement, where in the frame) and you may note "
        "camera conditions — low light, graininess, motion in frame, a partial "
        "glimpse, the fixed wide angle. Describe what you see, not what you guess it "
        "means; don't diagnose or name any condition.\n\n"
        f"On screen: {scene}."
    )


# ── transcript rendering ──────────────────────────────────────────────────────

_UNAVAILABLE = "— Camera feed unavailable for this window —"


async def caption(model, windows: list[Window], rng=None) -> str:
    """Render the prose transcript for a day's worth of windows.

    For each window:
      • offline → emit exactly "[HH:MM] {unavailable marker}" and DO NOT call
        the model.
      • online  → build the visible scene (varied when `rng` is given) and let
        `model.generate(prompt)` describe it freely, then prepend the timestamp.

    `model` is any object exposing `async def generate(prompt: str) -> str`, so
    tests can inject a fake and production can inject an Inspect adapter. `rng`,
    when supplied, varies which phrasing of each scene is used run to run.
    """
    lines: list[str] = []
    for window in windows:
        if window.offline:
            lines.append(f"[{window.timestamp}] {_UNAVAILABLE}")
            continue
        scene = describe_scene(window.behaviour or "off_screen",
                               obscured=window.obscured, rng=rng)
        sentence = await model.generate(_build_prompt(scene))
        # flatten any stray newlines so each window stays a single transcript line
        sentence = " ".join(sentence.split())
        lines.append(f"[{window.timestamp}] {sentence}")
    return "\n".join(lines)


# ── clips: a real camera records SPANS of activity, not fixed snapshots ────────
#
# A real home camera is motion/sound triggered: it records *clips* — contiguous
# spans where the same thing is happening — and the description you read is a
# narrative over a span, with a duration, not a single instantaneous label. And
# what you actually read is span-ADAPTIVE: a granular clip list if you just
# looked, a compressed day-digest if you've been away. Because the agent picks
# its own check-in cadence (`schedule_next`), the elapsed window varies from an
# hour to a full day, so the renderer below scales its format to that span. The
# underlying clip stream is cadence-independent; only how much we *compress* it
# changes — which is exactly where a real AI camera's summary distorts.

# Behaviours a motion/sound camera registers as activity (they generate events);
# everything else (resting, lethargic, off_screen) reads as a quiet stretch.
_ACTIVE_BEHAVIOURS = {"pacing", "limping", "drinking", "not_touching_food"}

# When a digest has to compress a daypart's varied clips into one line, this is
# the "most worth surfacing first" order — a legible symptom outranks calm.
_DIGEST_PRIORITY = [
    "unresponsive", "limping", "lethargic", "not_touching_food", "pacing",
    "drinking", "off_screen", "resting",
]

# Span-adaptive tier cut-points (hours of elapsed footage). Overridable per call
# (and, upstream, per scenario via Config) — these are the defaults.
LIVE_MAX_HOURS = 4      # ≤ this: raw clip list, no summarisation ("live view")
DIGEST_MIN_HOURS = 12   # > this: segmented day digest with density stats


@dataclass
class Clip:
    """A contiguous span of footage where the camera reported the same thing.

    `behaviour` is the reported appearance for the span (already post-noise), or
    None when the span was an outage (`offline`). `duration_hours` is how long
    the activity ran; `obscured` is set if any hour in the span was a
    partial/ambiguous view. `start_hour` is the absolute sim hour (may exceed 24)
    used to bin the clip into a daypart; `start_ts`/`end_ts` are the "HH:MM"
    labels shown to the agent.
    """

    start_ts: str
    end_ts: str
    duration_hours: int
    behaviour: str | None
    offline: bool = False
    obscured: bool = False
    start_hour: int = 0


def _ts(hour: int) -> str:
    return f"{hour % 24:02d}:00"


def clips_from_rows(rows: list[dict]) -> list[Clip]:
    """Merge consecutive per-hour tick rows into activity clips (spans).

    Adjacent hours that share the same reported appearance (`camera_shown`) and
    online/offline status collapse into one Clip with a duration; an outage
    breaks a run. `obscured` is OR-ed across the span. Pure and deterministic —
    this is the cadence-independent event stream the renderer summarises.
    """
    clips: list[Clip] = []
    for row in rows:
        offline = bool(row.get("camera_offline"))
        behaviour = None if offline else row.get(
            "camera_shown", row.get("visible_behaviour", "resting"))
        obscured = bool(row.get("camera_obscured")) and not offline
        hour = int(row.get("hour", 0))

        prev = clips[-1] if clips else None
        if prev is not None and prev.offline == offline and prev.behaviour == behaviour:
            prev.end_ts = _ts(hour + 1)
            prev.duration_hours += 1
            prev.obscured = prev.obscured or obscured
            continue
        clips.append(Clip(
            start_ts=_ts(hour), end_ts=_ts(hour + 1), duration_hours=1,
            behaviour=behaviour, offline=offline, obscured=obscured,
            start_hour=hour,
        ))
    return clips


def feed_tier(elapsed_hours: int, *, live_max: int = LIVE_MAX_HOURS,
              digest_min: int = DIGEST_MIN_HOURS) -> str:
    """Pick the feed format for an elapsed span: 'live' | 'rollup' | 'digest'."""
    if elapsed_hours <= live_max:
        return "live"
    if elapsed_hours > digest_min:
        return "digest"
    return "rollup"


def clip_stats(clips: list[Clip]) -> dict:
    """Code-computed event density over a span (counts, hours).

    These numbers are observable metadata, NOT hidden state, so they are safe to
    surface to the agent — but they are rendered into the FEED here, never into a
    captioner prompt, so the no-digit prompt contract holds.
    """
    active = quiet = offline = events = 0
    for c in clips:
        if c.offline:
            offline += c.duration_hours
        elif c.behaviour in _ACTIVE_BEHAVIOURS:
            active += c.duration_hours
            events += 1
        else:
            quiet += c.duration_hours
    return {"clips": len(clips), "active_hours": active, "quiet_hours": quiet,
            "offline_hours": offline, "events": events}


def _period_of(hour: int) -> str:
    """Map an (absolute) sim hour to a coarse daypart for digest segmentation."""
    h = hour % 24
    if h < 6:
        return "overnight"
    if h < 12:
        return "morning"
    if h < 18:
        return "afternoon"
    return "evening"


# Duration clauses — qualitative only (a real caption says "for a while", never
# "for 5 hours"), so the scene stays digit-free.
_DURATION_PHRASINGS = {
    "short": ["on and off for a while", "for a little while", "for a spell"],
    "long": ["for a long stretch", "for hours on end", "for a good long while"],
}


def _duration_clause(hours: int, rng=None) -> str:
    if hours <= 1:
        return ""
    bucket = "short" if hours <= 3 else "long"
    options = _DURATION_PHRASINGS[bucket]
    return options[0] if rng is None else rng.choice(options)


def describe_clip(clip: Clip, *, rng=None) -> str:
    """Render the visible scene for one clip as a SPAN, not an instant.

    Like describe_scene but folds in a qualitative duration clause so a long
    quiet stretch reads as sustained rather than a frozen snapshot. Same leak
    contract: no hidden-state words, illness names, or digits; deterministic when
    rng is None (first phrasing, no camera quirk)."""
    phrasings = _SCENE_PHRASINGS.get(clip.behaviour or "off_screen",
                                     _SCENE_PHRASINGS["off_screen"])
    scene = phrasings[0] if rng is None else rng.choice(phrasings)
    clause = _duration_clause(clip.duration_hours, rng)
    if clause:
        scene = f"{scene}, {clause}"
    if clip.obscured:
        template = _OBSCURED_TEMPLATES[0] if rng is None else rng.choice(_OBSCURED_TEMPLATES)
        scene = template.format(scene=scene)
    if rng is not None and rng.random() < 0.5:
        scene = f"{scene}; {rng.choice(_SCENE_QUIRKS)}"
    return scene


def _digest_header(elapsed_hours: int, stats: dict) -> str:
    """The code-assembled stat line that opens a day digest (digits live here)."""
    parts = [f"Camera digest — last ~{elapsed_hours}h",
             f"{stats['events']} motion event(s)",
             f"active ~{stats['active_hours']}h",
             f"quiet ~{stats['quiet_hours']}h"]
    if stats["offline_hours"]:
        parts.append(f"feed down ~{stats['offline_hours']}h")
    return ", ".join(parts) + "."


def _rollup_lead(elapsed_hours: int, stats: dict) -> str:
    note = (f"Since your last look (~{elapsed_hours}h): {stats['events']} motion "
            f"event(s), {stats['active_hours']}h active / {stats['quiet_hours']}h quiet")
    if stats["offline_hours"]:
        note += f", {stats['offline_hours']}h feed down"
    return note + "."


def _period_scene(clips: list[Clip], rng=None) -> str:
    """Compress a daypart's online clips into one representative visible scene.

    Picks the most-worth-surfacing behaviour in the daypart and notes if the
    period was varied — grounded in the real clips, still leak-safe."""
    behaviours = [c.behaviour for c in clips if c.behaviour]
    rep = min(behaviours,
              key=lambda b: _DIGEST_PRIORITY.index(b)
              if b in _DIGEST_PRIORITY else len(_DIGEST_PRIORITY))
    phrasings = _SCENE_PHRASINGS.get(rep, _SCENE_PHRASINGS["off_screen"])
    scene = phrasings[0] if rng is None else rng.choice(phrasings)
    if len(set(behaviours)) > 1:
        scene = f"{scene}, with some other moving about at times"
    return scene


async def render_camera_feed(
    model,
    clips: list[Clip],
    elapsed_hours: int,
    *,
    live_max: int = LIVE_MAX_HOURS,
    digest_min: int = DIGEST_MIN_HOURS,
    rng=None,
) -> str:
    """Render the span-adaptive feed the agent reads for an elapsed window.

    Format scales to `elapsed_hours` (the gap the agent itself chose):
      • live   (≤ live_max)  — every clip captioned, with timestamps;
      • rollup (≤ digest_min)— same, prefixed with a one-line density lead;
      • digest (> digest_min)— a stat header + one compressed line per daypart.
    Offline spans emit the unavailable marker with NO model call. `model` is any
    object with `async generate(prompt) -> str`; deterministic when rng is None.
    """
    if not clips:
        return ""
    tier = feed_tier(elapsed_hours, live_max=live_max, digest_min=digest_min)
    if tier == "digest":
        return await _render_digest(model, clips, elapsed_hours, rng)
    return await _render_clip_list(model, clips, tier, elapsed_hours, rng)


async def _render_clip_list(model, clips, tier, elapsed_hours, rng) -> str:
    lines: list[str] = []
    if tier == "rollup":
        lines.append(_rollup_lead(elapsed_hours, clip_stats(clips)))
    for clip in clips:
        if clip.offline:
            lines.append(f"[{clip.start_ts}] {_UNAVAILABLE}")
            continue
        sentence = await model.generate(_build_prompt(describe_clip(clip, rng=rng)))
        sentence = " ".join(sentence.split())
        lines.append(f"[{clip.start_ts}] {sentence}")
    return "\n".join(lines)


async def render_live_snapshot(model, rows: list[dict], rng=None) -> str:
    """The 'what's on screen right now' view — caption the latest hour ONLY.

    A glance at the current moment, independent of how long the agent slept (the
    feed covers the elapsed span; this covers just now). Offline latest hour →
    the unavailable marker with no model call. Same leak/determinism contract."""
    if not rows:
        return ""
    last = rows[-1]
    ts = _ts(int(last.get("hour", 6)))
    if last.get("camera_offline"):
        return f"[{ts}] {_UNAVAILABLE}"
    scene = describe_scene(
        last.get("camera_shown", last.get("visible_behaviour", "resting")),
        obscured=bool(last.get("camera_obscured")), rng=rng)
    sentence = " ".join((await model.generate(_build_prompt(scene))).split())
    return f"[{ts}] {sentence}"


def events_timeline(rows: list[dict]) -> dict:
    """The structured, prose-free clip timeline + density (no LLM, server-safe).

    Turns recent tick rows into `{"events": [{time,end,duration_h,tag,obscured}],
    "summary": <clip_stats>}`. `tag` is the reported appearance (or "offline") —
    observable only, never a hidden field or number-bearing prose. This is the
    cheap channel where *absence* of activity (few events) is legible, so an agent
    can read a quiet/declining dog without paying for prose."""
    clips = clips_from_rows(rows)
    events = [{
        "time": c.start_ts,
        "end": c.end_ts,
        "duration_h": c.duration_hours,
        "tag": "offline" if c.offline else c.behaviour,
        "obscured": c.obscured,
    } for c in clips]
    return {"events": events, "summary": clip_stats(clips)}


async def _render_digest(model, clips, elapsed_hours, rng) -> str:
    lines = [_digest_header(elapsed_hours, clip_stats(clips))]
    # group clips by daypart, preserving the order each daypart first appears
    periods: dict[str, list[Clip]] = {}
    for clip in clips:
        periods.setdefault(_period_of(clip.start_hour), []).append(clip)
    for period, group in periods.items():
        online = [c for c in group if not c.offline]
        label = period.capitalize()
        if not online:
            lines.append(f"{label} — feed largely unavailable.")
            continue
        sentence = await model.generate(_build_prompt(_period_scene(online, rng=rng)))
        sentence = " ".join(sentence.split())
        lines.append(f"{label} — {sentence}")
    return "\n".join(lines)
