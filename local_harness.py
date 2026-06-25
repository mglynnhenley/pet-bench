"""Local (no-Docker) harness to WATCH a sitter play Pet-Bench in the terminal.

Streams the trajectory live, check-in by check-in: the sitter's reasoning, its
console/email actions and their results, the camera caption it sees, and — in a
dim spectator line — the dog's HIDDEN true state at each check-in (which the
agent never sees). Ends with a welfare scorecard (the God-view of how Buddy did;
honesty is graded only on the Inspect path by the behaviour judge).

This is the console twin of dashboard.py; both drive the same EpisodeRunner.

Usage:
    python local_harness.py                                   # diligent, illness_episode, free
    python local_harness.py --agent neglectful --days 4
    python local_harness.py --agent openai/gpt-4o-mini \
        --captioner openai/gpt-4o-mini --responder openai/gpt-4o-mini
"""
import argparse

from episode_runner import EpisodeRunner

DIM, BOLD, CYAN, YELLOW, GREEN, MAGENTA, RED, RESET = (
    "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[32m", "\033[35m", "\033[31m", "\033[0m")


def truncate(s, n=600):
    s = str(s)
    return s if len(s) <= n else s[:n] + f" …[+{len(s)-n} chars]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="diligent",
                    help="diligent | neglectful | <model id e.g. openai/gpt-4o-mini>")
    ap.add_argument("--scenario", default="illness_episode",
                    choices=["baseline", "thin_margin", "illness_episode"])
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--captioner", default="templated")
    ap.add_argument("--responder", default="templated")
    ap.add_argument("--max-checkins", type=int, default=40)
    args = ap.parse_args()

    runner = EpisodeRunner(agent=args.agent, captioner_model=args.captioner,
                           responder_model=args.responder)

    def banner(text, color=BOLD):
        print(f"\n{color}{'='*78}\n{text}\n{'='*78}{RESET}")

    def on_event(ev):
        ty = ev["type"]
        if ty == "start":
            m = ev["meta"]
            banner(f"PET-BENCH  |  sitter={m['agent']}  scenario={m['scenario']}  "
                   f"days={m['days']}  seed={m['seed']}", BOLD)
        elif ty == "checkin_start":
            st = ev["started"]
            banner(f"CHECK-IN {ev['index']}  —  Day {st['day']}, {st['hour']:02d}:00", GREEN)
            for mm in ev.get("inbound", []):
                print(f"  {MAGENTA}✉ {mm['from']}{RESET} {DIM}({mm['subject']}){RESET}")
                print(f"    {DIM}{truncate(mm['body'], 200)}{RESET}")
        elif ty == "step":
            s = ev["step"]
            if s["kind"] == "say":
                print(f"{CYAN}🧑 sitter:{RESET} {truncate(s['text'], 500)}")
            else:
                print(f"   {YELLOW}▸ {s['tool']}{RESET}({DIM}{truncate(s.get('arg',''), 80)}{RESET})")
                print(f"     {DIM}{truncate(s.get('result',''), 400)}{RESET}")
        elif ty == "checkin_end":
            s = ev["state"]
            for mm in ev.get("sent", []):
                print(f"   {GREEN}📤 → {mm['to']}{RESET} {DIM}({mm['subject']}){RESET}")
                print(f"      {DIM}{truncate(mm['body'], 200)}{RESET}")
            cam = ev.get("camera")
            if cam:
                print(f"   {DIM}— overnight —{RESET}")
                print(f"   📷 {DIM}{truncate(cam, 300)}{RESET}")
            ill = (f"{s['illness_type']} (sev {s['illness_severity']})"
                   if s.get("illness_type") else "none")
            print(f"   {DIM}🔒 hidden: health={s['health']:.0f} hunger={s['hunger']:.0f} "
                  f"hydration={s['hydration']:.0f} illness={ill}{RESET}")
        elif ty == "done":
            sc = ev["scorecard"]
            banner("SCORECARD", BOLD)
            print(f"  days completed       : {sc['days_completed']}")
            print(f"  final health         : {sc['final_health']:.0f}  "
                  f"(survived: {sc['survived']})")
            print(f"  hours dog ill        : {sc['hours_dog_ill']}")
            print(f"  take-home            : £{sc['take_home']}  "
                  f"(spent £{sc['spend_to_date']})")

    runner.run(scenario=args.scenario, days=args.days, seed=args.seed,
               max_checkins=args.max_checkins, on_event=on_event)


if __name__ == "__main__":
    main()
