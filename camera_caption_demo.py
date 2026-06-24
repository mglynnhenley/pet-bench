"""Print REAL LLM camera captions via OpenRouter for one noisy day.

A standalone smoke test for the camera observable layer: it runs a genuinely
*limping* dog through a noisy feed (misread + obscured + offline) and prints the
captions the OpenRouter model actually emits, alongside the hidden ground truth —
so you can eyeball how unreliable the feed reads.

Usage:
    OPENROUTER_API_KEY=... python3 camera_caption_demo.py [model]
    # default model: openai/gpt-4o-mini
"""

import asyncio
import os
import random
import sys

import httpx

from petcare.config import Config
from petcare.world import new_world
from petcare.camera import caption
from petbench import _windows_from_rows

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-4o-mini"
KEY = os.environ["OPENROUTER_API_KEY"]


class OpenRouter:
    """Implements the captioner interface the camera expects: async generate()."""

    async def generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {KEY}",
                    "HTTP-Referer": "https://pet-bench.local",
                    "X-Title": "pet-bench camera demo",
                },
                json={"model": MODEL, "temperature": 1.0,
                      "messages": [{"role": "user", "content": prompt}]},
            )
        if r.status_code != 200:
            raise SystemExit(f"OpenRouter {r.status_code}: {r.text}")
        return r.json()["choices"][0]["message"]["content"].strip()


async def main():
    cfg = Config(illness_onset_prob=0, camera_offline_prob=0.15,
                 camera_misread_prob=0.4, camera_obscured_prob=0.3)
    w = new_world(cfg, seed=7)
    w.set_illness("limp", 3)        # truly limping the whole day
    w.advance(24)
    rows = w.tick_log
    window_hours = 4
    wins = _windows_from_rows(rows, window_hours=window_hours)

    transcript = await caption(OpenRouter(), wins, rng=random.Random())
    print(f"REAL LLM TRANSCRIPT (model={MODEL})\n" + "=" * 78)
    print(transcript)

    # Each window summarises its chunk by the chunk's FINAL hour (same rule as
    # _windows_from_rows), so derive the table from chunk[-1] to stay aligned.
    print("\nGROUND TRUTH (true behaviour = limping all day):\n" + "=" * 78)
    chunks = [rows[i:i + window_hours] for i in range(0, len(rows), window_hours)]
    for win, chunk in zip(wins, chunks):
        r = chunk[-1]
        kind = ("offline" if win.offline else
                "OBSCURED" if win.obscured else
                f"misread->{win.behaviour}" if r["camera_misread"] else "accurate")
        cam = "(down)" if win.offline else win.behaviour
        print(f"  {win.timestamp}  true={r['visible_behaviour']:<8}  "
              f"camera={cam:<10}  [{kind}]")


if __name__ == "__main__":
    asyncio.run(main())
