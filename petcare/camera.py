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
}

# ── thresholds (qualitative; only used to bin hidden state into coarse words) ─
_HUNGER_VISIBLE = 70.0      # at/above this the dog looks restless about food
_LOW_HEALTH_VISIBLE = 20.0  # at/below this the dog reads as run-down on camera
_LOW_HYDRATION = 30.0       # at/below this the dog is plausibly at the bowl


def derive_visible_behaviour(dog: Dog) -> str:
    """Map the dog's hidden state to ONE coarse, observable appearance label.

    Deterministic given the dog. The output is always a member of
    ALLOWED_BEHAVIOURS and is a qualitative word only — it never embeds any of
    the numeric hidden fields. Precedence is "most visible symptom first": a
    structural gait problem or a run-down dog reads on camera before mild
    hunger fidgeting does.
    """
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
