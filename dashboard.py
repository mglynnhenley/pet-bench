"""Pet-Bench trajectory explorer — a web GUI to watch and inspect episodes.

Run:
    python dashboard.py                 # serves on http://0.0.0.0:7861
    PORT=8000 python dashboard.py

Open the URL, pick an agent + scenario, click Run, and watch the sitter look
after Buddy check-in by check-in: its reasoning, its console/email actions, the
camera caption it sees — and, in a spectator-only panel, the dog's HIDDEN true
state at each step (which the agent never sees). Per-check-in charts + an honesty
scorecard summarise the run.

Agents: 'diligent' and 'neglectful' are instant scripted sitters (free). Pick a
frontier model (e.g. 'anthropic/claude-opus-4') from the dropdown — or paste any
OpenRouter model id into the custom field — to watch a real model play
(OPENROUTER_API_KEY).
"""
import copy
import json
import os
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from episode_runner import EpisodeRunner

RUNS_DIR = "/tmp/pet_runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Vendored front-end assets (GSAP) served locally so the demo runs offline.
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

SCENARIOS = ["baseline", "thin_margin", "illness_episode"]

app = FastAPI(title="Pet-Bench Explorer")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
RUNS: dict[str, dict] = {}


class RunReq(BaseModel):
    agent: str = "diligent"
    scenario: str = "illness_episode"
    days: int = 5
    captioner: str = "templated"
    responder: str = "templated"
    seed: int = 0


def _worker(run_id: str, req: RunReq):
    rec = RUNS[run_id]
    t = rec["trajectory"]
    try:
        runner = EpisodeRunner(agent=req.agent, captioner_model=req.captioner,
                               responder_model=req.responder)

        def on_event(ev):
            with rec["lock"]:
                ty = ev["type"]
                if ty == "checkin_start":
                    t["checkins"].append({"index": ev["index"], "started": ev["started"],
                                          "brief": ev["brief"], "inbound": ev["inbound"],
                                          "steps": [], "state": None, "camera": None, "sent": []})
                elif ty == "step":
                    if t["checkins"]:
                        t["checkins"][-1]["steps"].append(ev["step"])
                elif ty == "checkin_end":
                    if t["checkins"]:
                        t["checkins"][-1]["state"] = ev["state"]
                        t["checkins"][-1]["camera"] = ev["camera"]
                        t["checkins"][-1]["sent"] = ev.get("sent", [])
                    rec["progress"] = ev["index"] + 1
                elif ty == "done":
                    t["scorecard"] = ev["scorecard"]

        full = runner.run(scenario=req.scenario, days=req.days, seed=req.seed,
                          on_event=on_event)
        with rec["lock"]:
            rec["trajectory"] = full
        rec["status"] = "done"
        with open(os.path.join(RUNS_DIR, f"{run_id}.json"), "w") as f:
            json.dump({"meta": rec["meta"], "trajectory": full}, f)
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"


@app.post("/api/run")
def start_run(req: RunReq):
    run_id = uuid.uuid4().hex[:8]
    created = time.time()
    meta = {"agent": req.agent, "scenario": req.scenario, "days": req.days,
            "captioner": req.captioner, "responder": req.responder, "seed": req.seed,
            "id": run_id, "created": created}
    RUNS[run_id] = {"status": "running", "progress": 0, "error": None,
                    "lock": threading.Lock(), "meta": meta,
                    "trajectory": {"meta": meta, "checkins": [], "scorecard": None}}
    threading.Thread(target=_worker, args=(run_id, req), daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/run/{run_id}")
def get_run(run_id: str):
    if run_id not in RUNS:
        path = os.path.join(RUNS_DIR, f"{run_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            return {"status": "done", "meta": d["meta"], "trajectory": d["trajectory"]}
        raise HTTPException(404, "no such run")
    r = RUNS[run_id]
    with r["lock"]:
        traj = copy.deepcopy(r.get("trajectory"))
    return {"status": r["status"], "progress": r.get("progress", 0), "error": r.get("error"),
            "meta": r["meta"], "trajectory": traj}


@app.get("/api/runs")
def list_runs():
    seen = {}
    for rid, r in RUNS.items():
        seen[rid] = {"id": rid, "status": r["status"], **r["meta"]}
    for fn in os.listdir(RUNS_DIR):
        if fn.endswith(".json"):
            rid = fn[:-5]
            if rid in seen:
                continue
            try:
                with open(os.path.join(RUNS_DIR, fn)) as f:
                    meta = json.load(f).get("meta", {})
                seen[rid] = {"id": rid, "status": "done", **meta}
            except Exception:
                pass
    return {"runs": sorted(seen.values(), key=lambda x: x.get("created", 0))}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = r"""
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Petbench</title>
<style>
  /* ── Editorial light theme — Ramp surfaces + Stripe Press serif headings.
     Restrained palette, hairline borders, no gradients/glow. The chain of
     thought is the hero: a comfortable serif reading column you skim top-down. */
  :root{
    --paper:#fbfbf9; --surface:#ffffff; --raise:#f6f5f1;
    --ink:#191815; --ink2:#5b574e; --ink3:#928d81;
    --line:#ecebe4; --line2:#dbd9cd;
    --accent:#1a1a17; --mark:#c8e000;          /* near-black UI ink + a single lime mark */
    --good:#1d7a4d; --bad:#c03a23; --warn:#a96a00; --info:#2f6f9e;
    --note:#fbf6e8; --note-line:#ecdcab; --note-ink:#7a6420;
    --r:10px; --rs:7px; --measure:70ch;
    --sans:'Inter','Inter var',system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --serif:'Iowan Old Style','Charter','Palatino Linotype','Georgia',serif;
    --mono:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;
  }
  *{box-sizing:border-box}
  ::selection{background:#e8f08a}
  html{scroll-behavior:smooth}
  body{margin:0;font:15px/1.6 var(--sans);color:var(--ink);background:var(--paper);
       -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
  ::-webkit-scrollbar{width:10px;height:10px}
  ::-webkit-scrollbar-thumb{background:#d8d6cc;border:3px solid transparent;border-radius:8px;background-clip:padding-box}
  ::-webkit-scrollbar-thumb:hover{background:#c3c0b4;background-clip:padding-box}
  a{color:inherit}

  header{padding:15px 28px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:14px;
         background:var(--surface);position:sticky;top:0;z-index:40}
  header h1{font:600 19px/1 var(--serif);margin:0;letter-spacing:-.01em}
  header h1 .dot{color:var(--ink3);font-weight:400}
  header .sub{color:var(--ink3);font-size:13px}
  .wrap{display:flex;height:calc(100vh - 55px)}

  /* ── sidebar ──────────────────────────────────────────────────────────── */
  .side{width:288px;border-right:1px solid var(--line);padding:22px 20px;overflow:auto;flex-shrink:0;background:var(--surface)}
  .side .eyebrow{font:600 11px/1 var(--sans);letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);margin:0 0 14px}
  label{display:block;font-size:12px;color:var(--ink2);margin:15px 0 5px;font-weight:600}
  select,input,button{width:100%;background:var(--surface);color:var(--ink);border:1px solid var(--line2);
        border-radius:var(--rs);padding:9px 11px;font:14px/1.4 var(--sans);transition:border-color .12s,box-shadow .12s}
  select,input{appearance:none}
  select{background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'><path d='M3 4.5l3 3 3-3' fill='none' stroke='%23928d81' stroke-width='1.5' stroke-linecap='round'/></svg>");
        background-repeat:no-repeat;background-position:right 11px center;padding-right:30px;cursor:pointer}
  #agentCustom{margin-top:7px}
  .hint{display:block;font-size:11px;color:var(--ink3);margin-top:5px;line-height:1.4}
  select:hover,input:hover{border-color:#c7c4b8}
  select:focus,input:focus{outline:none;border-color:var(--ink);box-shadow:0 0 0 3px rgba(25,24,21,.08)}
  button{background:var(--accent);color:#fff;border:1px solid var(--accent);font-weight:600;cursor:pointer;
        margin-top:20px;padding:11px}
  button:hover{background:#000}
  button:disabled{opacity:.5;cursor:default}

  .runs{margin-top:26px;border-top:1px solid var(--line);padding-top:18px}
  .runitem{display:flex;align-items:center;gap:9px;padding:8px 9px;border-radius:var(--rs);margin:0 -9px;
        cursor:pointer;font-size:13px;transition:background .12s}
  .runitem b{font-weight:600;flex-shrink:0;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .runitem .rmeta{color:var(--ink3);font-size:11.5px;margin-left:auto;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:right}
  .rdot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
  .rdot.b-done{background:var(--good)} .rdot.b-err{background:var(--bad)}
  .rdot.b-run{background:var(--info);animation:pulse 1.1s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
  .runitem:hover{background:var(--raise)}
  .runitem.active{background:var(--raise);box-shadow:inset 2px 0 0 var(--ink)}

  /* ── main column ──────────────────────────────────────────────────────── */
  .main{flex:1;overflow:auto;padding:0}
  .col{max-width:1080px;margin:0 auto;padding:30px 40px 120px}
  .empty{color:var(--ink3);text-align:center;margin-top:90px;font:17px/1.6 var(--serif)}

  .runhead{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:4px}
  .runhead h2{font:600 26px/1.15 var(--serif);margin:0;letter-spacing:-.015em}
  .pill{font-size:12px;color:var(--ink2);background:var(--raise);border:1px solid var(--line);
        border-radius:20px;padding:2px 10px;white-space:nowrap}
  .live-tag{font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--info);
        display:inline-flex;align-items:center;gap:6px}
  .live-tag::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--info);animation:pulse 1.1s infinite}

  /* summary — two labelled groups, plain-language so the metrics are legible */
  .summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px;margin:22px 0 6px}
  .glab{font:600 11px/1 var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--ink3);margin:0 0 8px}
  .glab .cap{font-weight:500;letter-spacing:0;text-transform:none;color:var(--ink3)}
  /* stat strip — flat figures, semantic colour on the number only */
  .stats{display:flex;flex-wrap:wrap;gap:0;border:1px solid var(--line);border-radius:var(--r);
        background:var(--surface);overflow:hidden}
  .stat{flex:1;min-width:104px;padding:13px 16px;border-right:1px solid var(--line)}
  .stat:last-child{border-right:none}
  .stat .k{font-size:11px;color:var(--ink3);font-weight:600;letter-spacing:.02em}
  .stat .v{font:600 21px/1.1 var(--sans);margin-top:5px;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
  .v.good{color:var(--good)} .v.bad{color:var(--bad)} .v.warn{color:var(--warn)}

  /* vitals — minimal sparklines */
  .vitals{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0 6px}
  .vital .vh{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:2px}
  .vital .vh b{font-size:12px;font-weight:600;color:var(--ink2)}
  .vital .vh span{font:600 13px/1 var(--sans);font-variant-numeric:tabular-nums;color:var(--ink)}

  /* day jump nav — sticky, quiet */
  .daynav{position:sticky;top:0;z-index:10;display:flex;gap:6px;flex-wrap:wrap;padding:12px 0;margin-top:8px;
        background:linear-gradient(var(--paper) 70%,rgba(251,251,249,0))}
  .chip{font-size:12px;font-weight:600;color:var(--ink2);background:var(--surface);border:1px solid var(--line2);
        border-radius:20px;padding:4px 11px;cursor:pointer;font-variant-numeric:tabular-nums;transition:.12s}
  .chip:hover{border-color:var(--ink);color:var(--ink)}
  .chip.harm{color:var(--bad);border-color:#e7c3b8}
  .chip .h{color:var(--ink3);font-weight:500;margin-left:3px}

  /* ── a check-in: the reading unit ─────────────────────────────────────── */
  .daysep{font:600 13px/1 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);
        margin:38px 0 0;padding-bottom:10px;border-bottom:1px solid var(--line)}
  .ci{padding:22px 0;border-bottom:1px solid var(--line)}
  .ci-head{display:flex;align-items:center;gap:12px;margin-bottom:13px}
  .ci-head .when{font:600 16px/1 var(--serif)}
  .ci-head .saw{font-size:12.5px;color:var(--ink3)}
  .ci-head .saw b{color:var(--ink2);font-weight:600}
  /* spectator ground-truth pill — what was REALLY going on (agent can't see it) */
  .status{margin-left:auto;display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;
        border:1px solid var(--line2);border-radius:20px;padding:3px 10px 3px 8px}
  .status::before{content:"";width:7px;height:7px;border-radius:50%}
  .status .tl{font-weight:500;color:var(--ink3);font-size:10.5px;letter-spacing:.04em;text-transform:uppercase}
  .status.ok{color:var(--good)} .status.ok::before{background:var(--good)}
  .status.ill{color:var(--bad);border-color:#e7c3b8;background:#fdf4f1} .status.ill::before{background:var(--bad)}

  /* reasoning — the chain of thought, set as editorial prose */
  .reason{max-width:var(--measure)}
  .reason .think{font:16.5px/1.72 var(--serif);color:var(--ink);margin:0 0 13px;white-space:pre-wrap}
  .reason .think:last-child{margin-bottom:0}
  .nothing{color:var(--ink3);font-style:italic;font-size:14px}

  /* private notepad — a candid aside, tasteful not gimmicky */
  .note{max-width:var(--measure);background:var(--note);border:1px solid var(--note-line);border-left:3px solid #d9c372;
        border-radius:var(--rs);padding:11px 15px;margin:14px 0}
  .note .nlab{font:600 10.5px/1 var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--note-ink);
        margin-bottom:7px;display:flex;align-items:center;gap:8px}
  .note .nlab .ptag{font-weight:500;letter-spacing:0;text-transform:none;color:var(--ink3);
        border:1px solid var(--note-line);border-radius:5px;padding:1px 6px;font-size:10px}
  .note .ntext{font:15px/1.65 var(--serif);color:#5e4f25;white-space:pre-wrap}

  /* "Did" — one-line plain summary of what the agent actually did this turn */
  .didline{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin:15px 0 0}
  .didline .dlab{font:600 11px/1 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);flex-shrink:0}
  .vtag{font-size:12px;color:var(--ink2);background:var(--raise);border:1px solid var(--line);
        border-radius:6px;padding:2px 8px}
  /* one consolidated per-turn disclosure: console log, camera feed, this turn's mail */
  .dsub{font:600 10.5px/1 var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);margin:16px 0 8px}
  .dsub:first-child{margin-top:0}
  /* actions — a compact monospace ledger, clearly secondary to the prose */
  .ledger{border:1px solid var(--line);border-radius:var(--rs);overflow:hidden;background:var(--surface)}
  .act{display:flex;align-items:flex-start;gap:12px;padding:9px 14px;border-top:1px solid var(--line);cursor:default}
  .act:first-child{border-top:none}
  .act .verb{font-size:13px;font-weight:600;color:var(--ink);flex-shrink:0;width:172px}
  .act .cmd{font:12px/1.5 var(--mono);color:var(--ink2);flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .act.open .cmd{white-space:pre-wrap}
  .act .res{display:none;font:11.5px/1.5 var(--mono);color:var(--ink3);white-space:pre-wrap;margin-top:7px;max-height:220px;
        overflow:auto;background:var(--raise);border:1px solid var(--line);border-radius:6px;padding:8px 10px}
  .act.open .res{display:block}
  .act .toggle{margin-left:auto;font-size:11px;font-weight:600;color:var(--ink3);cursor:pointer;flex-shrink:0;user-select:none}
  .act .toggle::after{content:"result"} .act.open .toggle::after{content:"hide"}
  .act .toggle:hover{color:var(--ink)}

  /* a streaming indicator */
  .typing{display:flex;gap:6px;padding:10px 0 2px}
  .typing i{width:6px;height:6px;border-radius:50%;background:var(--ink3);animation:tbl 1.2s infinite}
  .typing i:nth-child(2){animation-delay:.18s} .typing i:nth-child(3){animation-delay:.36s}
  @keyframes tbl{0%,60%,100%{opacity:.2}30%{opacity:1}}

  /* ── collapsible sections (camera / inbox / day table) ────────────────── */
  .sect{margin:16px 0 0}
  .shead{cursor:pointer;user-select:none;display:flex;align-items:center;gap:8px;
        font:600 13px/1 var(--sans);color:var(--ink2);margin:0}
  .shead .chev{font-size:9px;color:var(--ink3);transition:transform .15s}
  .sect.closed .shead .chev{transform:rotate(-90deg)}
  .sect.closed .sbody{display:none}
  .sbody{margin-top:11px}

  .cam{white-space:pre-wrap;background:var(--raise);border:1px solid var(--line);border-radius:var(--rs);
       padding:12px 15px;color:var(--ink2);font:13px/1.6 var(--serif);font-style:italic}

  /* inbox — quiet rows, click to expand */
  .inbox{border:1px solid var(--line);border-radius:var(--rs);overflow:hidden;background:var(--surface)}
  .mitem{border-bottom:1px solid var(--line)}
  .mitem:last-child{border-bottom:none}
  .mrow{display:flex;align-items:center;gap:12px;padding:10px 14px;cursor:pointer;transition:background .12s}
  .mrow:hover{background:var(--raise)}
  .mitem.open .mrow{background:var(--raise)}
  .mrow .dir{font-size:11px;font-weight:600;flex-shrink:0;width:30px;color:var(--ink3);letter-spacing:.03em}
  .mitem.sent .mrow .dir{color:var(--good)}
  .mrow .who{font-weight:600;font-size:12.5px;color:var(--ink);flex-shrink:0;max-width:200px;overflow:hidden;
        text-overflow:ellipsis;white-space:nowrap}
  .mrow .subj{flex:1;min-width:0;font-size:13px;color:var(--ink2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .mitem.open .mrow .subj{white-space:normal;color:var(--ink)}
  .mrow .mday{font-size:11px;color:var(--ink3);flex-shrink:0;font-variant-numeric:tabular-nums}
  .mbody{display:none;padding:0 16px 15px 44px;font:13px/1.65 var(--sans);color:var(--ink2);white-space:pre-wrap}
  .mitem.open .mbody{display:block}

  /* day table — clean ground truth */
  .dtab{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
  .dtab th{text-align:right;color:var(--ink3);font-weight:600;padding:7px 12px;border-bottom:1px solid var(--line2)}
  .dtab th:first-child{text-align:left}
  .dtab td{text-align:right;padding:8px 12px;border-bottom:1px solid var(--line)}
  .dtab td:first-child{text-align:left;font-weight:600}
  .dtab tbody tr:hover{background:var(--raise)}

  @media (max-width:720px){.vitals{grid-template-columns:repeat(2,1fr)}.act .verb{width:auto}}
  @media (prefers-reduced-motion:reduce){*{animation:none!important;scroll-behavior:auto}}
</style></head><body>
<header><h1>Petbench<span class="dot"> / explorer</span></h1>
  <span class="sub">read how an agent reasoned, acted, and what was really happening to Buddy</span></header>
<div class="wrap">
  <div class="side">
    <p class="eyebrow">New run</p>
    <label>Agent</label>
    <select id="agent">
      <optgroup label="scripted (free)">
        <option value="diligent">diligent</option>
        <option value="neglectful">neglectful</option>
      </optgroup>
      <optgroup label="frontier (live)">
        <option value="anthropic/claude-opus-4">anthropic/claude-opus-4</option>
        <option value="anthropic/claude-sonnet-4">anthropic/claude-sonnet-4</option>
        <option value="openai/gpt-4o">openai/gpt-4o</option>
        <option value="google/gemini-2.5-pro">google/gemini-2.5-pro</option>
      </optgroup>
      <optgroup label="small (live)">
        <option value="openai/gpt-4o-mini">openai/gpt-4o-mini</option>
      </optgroup>
    </select>
    <input id="agentCustom" placeholder="…or any OpenRouter model id">
    <span class="hint">A custom id overrides the dropdown (e.g. anthropic/claude-opus-4.1).</span>
    <label>Scenario</label>
    <select id="scenario">
      <option value="illness_episode">illness_episode</option>
      <option value="baseline">baseline</option>
      <option value="thin_margin">thin_margin</option>
    </select>
    <label>Episode length (days)</label><input id="days" type="number" value="5" min="1" max="14">
    <label>Camera captioner</label>
    <select id="captioner">
      <option value="templated">templated (free)</option>
      <option value="openai/gpt-4o-mini">gpt-4o-mini (live)</option>
    </select>
    <label>Counterparty emails</label>
    <select id="responder">
      <option value="templated">templated (free)</option>
      <option value="openai/gpt-4o-mini">gpt-4o-mini (live)</option>
    </select>
    <label>Seed</label><input id="seed" type="number" value="0" min="0">
    <button id="runbtn" onclick="startRun()">Run episode</button>
    <div class="runs" id="runs"></div>
  </div>
  <div class="main" id="main"><div class="col"><div class="empty">Choose an agent and press Run to watch it look after Buddy.</div></div></div>
</div>

<script>
let active=null, poll=null, following=true, lastSig='';
const openMails=new Set();   // email keys currently expanded (persist across live re-renders)
const openActs=new Set();    // action results currently expanded
const openSects=new Set();   // section keys the user opened (sections start folded)
function val(id){return document.getElementById(id).value}
function escA(s){return (s==null?'':''+s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function esc(s){return (s==null?'':''+s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}

// One delegated handler: expand email rows, expand action results, fold sections.
document.addEventListener('click',e=>{
  const row=e.target.closest('.mrow');
  if(row){ const it=row.closest('.mitem'), k=it.dataset.mkey;
    if(it.classList.toggle('open')) openMails.add(k); else openMails.delete(k); return; }
  const tg=e.target.closest('.act .toggle');
  if(tg){ const a=tg.closest('.act'), k=a.dataset.akey;
    if(a.classList.toggle('open')) openActs.add(k); else openActs.delete(k); return; }
  const sh=e.target.closest('.shead');
  if(sh){ const se=sh.closest('.sect'), k=se.dataset.skey;
    if(se.classList.toggle('closed')) openSects.delete(k); else openSects.add(k); }
});

// A foldable section. Folded by default to keep the transcript skimmable; the
// open set persists across the signature-driven re-render during a live run.
function section(key,title,body){
  const off=!openSects.has(key);
  return `<div class="sect ${off?'closed':''}" data-skey="${escA(key)}">`+
    `<div class="shead"><span class="chev">▾</span>${title}</div>`+
    `<div class="sbody">${body}</div></div>`;
}

function mailRow(m,i,pfx){
  const sent=m.dir==='sent';
  const who=sent?esc(m.to||''):esc(m.from||'');
  const key=(pfx||'')+'|'+(m.day==null?'':m.day)+'|'+(sent?'s':'r')+'|'+(m.to||m.from||'')+'|'+(m.subject||'')+'|'+i;
  const open=openMails.has(key);
  const body=esc(m.body||'')||'<i style="color:var(--ink3)">(empty message)</i>';
  return `<div class="mitem ${sent?'sent':''} ${open?'open':''}" data-mkey="${escA(key)}">`+
    `<div class="mrow"><span class="dir">${sent?'SENT':'IN'}</span>`+
    `<span class="who">${who}</span>`+
    `<span class="subj">${esc(m.subject||'(no subject)')}</span>`+
    `<span class="mday">${m.day!=null?'day '+m.day:''}</span></div>`+
    `<div class="mbody">${body}</div></div>`;
}
function inbox(list,pfx){ return `<div class="inbox">`+list.map((m,i)=>mailRow(m,i,pfx)).join('')+`</div>`; }

async function startRun(){
  const body={agent:(val('agentCustom').trim()||val('agent')),scenario:val('scenario'),days:+val('days'),
    captioner:val('captioner'),responder:val('responder'),seed:+val('seed')};
  const btn=document.getElementById('runbtn'); btn.disabled=true; btn.textContent='running…';
  const {run_id}=await (await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  active=run_id; following=true; lastSig='';
  await refreshRuns(); startPolling(run_id);
}
function startPolling(id){ if(poll) clearInterval(poll); tick(id); poll=setInterval(()=>tick(id),1200); }
async function tick(id){
  const d=await (await fetch('/api/run/'+id)).json(); render(id,d);
  if(d.status!=='running'){ clearInterval(poll); poll=null;
    const b=document.getElementById('runbtn'); b.disabled=false; b.textContent='Run episode'; refreshRuns(); }
}
async function selectRun(id){
  active=id; following=true; lastSig='';
  if(poll){clearInterval(poll);poll=null;}
  const d=await (await fetch('/api/run/'+id)).json(); render(id,d);
  if(d.status==='running') startPolling(id); else refreshRuns();
}
async function refreshRuns(){
  const {runs}=await (await fetch('/api/runs')).json();
  document.getElementById('runs').innerHTML='<p class="eyebrow">Runs</p>'+runs.slice().reverse().map(r=>{
    const cls=r.status==='done'?'b-done':r.status==='error'?'b-err':'b-run';
    return `<div class="runitem ${r.id===active?'active':''}" onclick="selectRun('${r.id}')">
      <span class="rdot ${cls}"></span><b>${esc(r.agent)}</b>
      <span class="rmeta">${esc(r.scenario)} · ${r.days}d</span></div>`;}).join('');
}

function render(id,d){
  const main=document.getElementById('main');
  if(d.status==='error'){main.innerHTML=`<div class="col"><div class="empty" style="color:var(--bad)">Error — ${esc(d.error)}</div></div>`;lastSig='err';return;}
  const t=d.trajectory;
  if(!t||!t.checkins){main.innerHTML='<div class="col"><div class="empty">Starting…</div></div>';lastSig='';return;}
  const cks=t.checkins, sc=t.scorecard, running=d.status==='running';
  const lastSteps=cks.length?cks[cks.length-1].steps.length:0;
  const sig=d.status+'|'+cks.length+'|'+lastSteps+'|'+(sc?1:0);
  if(sig===lastSig) return;
  lastSig=sig;

  const sb=main.scrollTop;                       // preserve scroll across re-render
  const meta=d.meta||{};
  let html='<div class="col">';
  html+=`<div class="runhead"><h2>${esc(meta.agent||'agent')}</h2>`+
    `<span class="pill">${esc(meta.scenario||'')}</span>`+
    `<span class="pill">seed ${meta.seed??0}</span>`+
    `<span class="pill">${meta.days||cks.length}-day episode</span>`+
    (running?`<span class="live-tag">live · ${cks.length} check-in${cks.length===1?'':'s'}</span>`:'')+`</div>`;
  if(sc) html+=stats(sc);
  html+=vitals(cks);
  html+=daynav(cks);
  html+=transcript(cks,running);
  html+=dayStatus(t);
  html+=allEmails(t);
  html+='</div>';
  main.innerHTML=html;

  // jump-nav → scroll to the matching check-in
  document.querySelectorAll('.chip').forEach(ch=>ch.onclick=()=>{
    following=false; const el=document.getElementById('ci-'+ch.dataset.i);
    if(el) el.scrollIntoView({block:'start'});
  });
  // detect manual scroll-up → stop auto-following the live tail
  main.onscroll=()=>{ if(running && main.scrollTop < main.scrollHeight-main.clientHeight-80) following=false; };

  if(running && following){ main.scrollTop=main.scrollHeight; }
  else{ main.scrollTop=sb; }
}

function stats(sc){
  const s=(k,v,cls='')=>`<div class="stat"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`;
  const h=sc.final_health==null?'—':Math.round(sc.final_health);
  const welfare='<div class="stats">'+
    s('Final health', h, sc.survived?(h>=60?'good':'warn'):'bad')+
    s('Hours ill', sc.hours_dog_ill, sc.hours_dog_ill>0?'warn':'')+
    s('Take-home', '£'+(sc.take_home??'—'))+'</div>';
  const honesty='<div class="stats">'+
    s('False completion', sc.false_completion, sc.false_completion>0?'bad':'good')+
    s('Symptom dismissal', sc.symptom_dismissal, sc.symptom_dismissal>0?'bad':'good')+
    s('Concealed uncertainty', sc.concealed_uncertainty, sc.concealed_uncertainty>0?'bad':'good')+'</div>';
  return '<div class="summary">'+
    `<div><div class="glab">How Buddy did</div>${welfare}</div>`+
    `<div><div class="glab">Honesty to the owner <span class="cap">— times its claims contradicted reality · 0 is good</span></div>${honesty}</div>`+
    '</div>';
}

function vitals(cks){
  const states=cks.map(c=>c.state).filter(Boolean);
  if(!states.length) return '';
  const series=[['health','Health','#1d7a4d'],['hunger','Hunger','#a96a00'],
    ['hydration','Hydration','#2f6f9e'],['take_home','Take-home £','#191815']];
  return '<div class="glab" style="margin-top:24px">Buddy across the episode <span class="cap">— ground truth, per check-in</span></div>'+
    '<div class="vitals">'+series.map(([k,lab,col])=>{
    const data=states.map(s=>s[k]||0), last=data[data.length-1];
    return `<div class="vital"><div class="vh"><b>${lab}</b><span>${Math.round(last)}</span></div>${spark(data,col)}</div>`;
  }).join('')+'</div>';
}
function spark(data,col){
  const W=260,H=44,p=4; const mn=Math.min(...data,0),mx=Math.max(...data,1);
  const x=i=>p+i*(W-2*p)/Math.max(data.length-1,1), y=v=>H-p-(v-mn)/(mx-mn||1)*(H-2*p);
  const pts=data.map((v,i)=>x(i)+','+y(v)).join(' ');
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" preserveAspectRatio="none">
    <polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${x(data.length-1)}" cy="${y(data[data.length-1])}" r="2.4" fill="${col}"/></svg>`;
}

function daynav(cks){
  if(!cks.length) return '';
  // one chip per day (jumps to that day's first check-in); flag a day where Buddy was ill
  const seen=new Set(); let chips='';
  cks.forEach((c,i)=>{
    const day=(c.started||{}).day;
    if(seen.has(day)) return; seen.add(day);
    const harm=cks.some(x=>(x.started||{}).day===day && x.state && x.state.illness_severity>0);
    const n=cks.filter(x=>(x.started||{}).day===day).length;
    chips+=`<span class="chip ${harm?'harm':''}" data-i="${i}">Day ${day}<span class="h">·${n}</span></span>`;
  });
  return '<div class="daynav">'+chips+'</div>';
}

// The transcript — every check-in stacked for top-to-bottom skimming, grouped by day.
function transcript(cks,running){
  if(!cks.length) return '<div class="nothing" style="margin-top:24px">The agent is settling in…</div>';
  let html='', curDay=null;
  cks.forEach((c,i)=>{
    const st=c.started||{};
    if(st.day!==curDay){ curDay=st.day; html+=`<div class="daysep">Day ${st.day}</div>`; }
    html+=checkin(c,i,running && i===cks.length-1);
  });
  return html;
}

function checkin(c,i,follow){
  const st=c.started||{}, s=c.state;
  let saw = s && s.camera_offline ? 'camera offline' : ((s && s.visible_behaviour) || 'resting').replace(/_/g,' ');
  if((s&&s.visible_behaviour==='resting') && s.ate_recently) saw='eating';
  // header: when · what the camera showed the agent · what was REALLY true (spectator)
  let h=`<article class="ci" id="ci-${i}"><div class="ci-head">`+
    `<span class="when">Day ${st.day} · ${String(st.hour).padStart(2,'0')}:00</span>`+
    `<span class="saw">camera showed <b>${esc(saw)}</b></span>`;
  if(s) h+= s.illness_type
    ? `<span class="status ill"><span class="tl">truth</span> ${esc(s.illness_type)} sev ${s.illness_severity}</span>`
    : `<span class="status ok"><span class="tl">truth</span> well</span>`;
  h+='</div>';

  const says=c.steps.filter(x=>x.kind==='say');
  const notes=c.steps.filter(x=>x.tool==='scratchpad');
  const acts=c.steps.filter(x=>x.kind!=='say' && x.tool!=='scratchpad');

  // reasoning — the chain of thought, as prose (the thing you skim)
  if(says.length) h+='<div class="reason">'+says.map(x=>`<p class="think">${esc(x.text)}</p>`).join('')+'</div>';
  else if(!c.steps.length) h+='<div class="nothing">No activity recorded.</div>';
  // private notepad inline (candid CoT, still covertly graded)
  notes.forEach(n=>{
    h+=`<div class="note"><div class="nlab">Private notepad<span class="ptag">agent believes this is unmonitored</span></div>`+
       `<div class="ntext">${esc(n.arg||n.result||'')}</div></div>`;
  });

  // one-line "Did" summary — collapse consecutive duplicate verbs
  if(acts.length){
    const verbs=acts.map(a=>actionLabel(a.tool,a.arg)).filter((v,k,arr)=>v!==arr[k-1]);
    h+=`<div class="didline"><span class="dlab">Did</span>${verbs.map(v=>`<span class="vtag">${esc(v)}</span>`).join('')}</div>`;
  }

  // ONE consolidated disclosure for everything secondary: console log, camera, mail
  let det='';
  if(acts.length){ det+='<div class="dsub">Console log</div><div class="ledger">'+acts.map((a,j)=>{
      const key='ci'+i+'-'+j;
      return actMarkup(key,actionLabel(a.tool,a.arg),fmtCmd(a.tool,a.arg),fmtResult(a.result));
    }).join('')+'</div>'; }
  if(c.camera) det+='<div class="dsub">Camera feed</div><div class="cam">'+esc(c.camera)+'</div>';
  if(c.inbound&&c.inbound.length){ const ms=c.inbound.map(mm=>({dir:'in',from:mm.from,subject:mm.subject,body:mm.body}));
    det+=`<div class="dsub">Inbox at this check-in (${ms.length})</div>`+inbox(ms,'in'+i); }
  if(c.sent&&c.sent.length){ const ms=c.sent.map(mm=>({dir:'sent',to:mm.to,subject:mm.subject,body:mm.body}));
    det+=`<div class="dsub">Emails sent (${ms.length})</div>`+inbox(ms,'out'+i); }
  if(det) h+=section('det-'+i,'Details — console, camera & mail',det);

  if(follow) h+='<div class="typing"><i></i><i></i><i></i></div>';
  h+='</article>';
  return h;
}

function actMarkup(key,ai,cmd,res){
  const open=openActs.has(key), hasRes=(''+res).trim().length>0;
  const resHtml=hasRes?`<div class="res">${esc(res)}</div>`:'';
  const tog=hasRes?`<span class="toggle"></span>`:'';
  return `<div class="act ${open?'open':''}" data-akey="${escA(key)}">`+
    `<span class="verb">${esc(ai)}</span>`+
    `<div style="flex:1;min-width:0"><span class="cmd">${esc(cmd)}</span>${resHtml}</div>${tog}</div>`;
}

// ground-truth status at the end of each completed day (spectator only)
function dayStatus(t){
  const dl=(t.daily_log||[]).filter(Boolean);
  if(!dl.length) return '';
  const num=v=>v==null?'—':Math.round(v);
  const rows=dl.map(s=>{
    const ill=s.illness_type?`${esc(s.illness_type)} (sev ${s.illness_severity})`:'healthy';
    const illc=(s.illness_severity||0)>0?'var(--bad)':'var(--good)';
    return `<tr><td>Day ${s.day}</td><td>${num(s.health)}</td><td>${num(s.hunger)}</td>`+
      `<td>${num(s.hydration)}</td><td style="color:${illc}">${ill}</td>`+
      `<td>${s.food_stock??'—'}</td><td>${s.med_stock??'—'}</td><td>£${s.take_home??'—'}</td></tr>`;
  }).join('');
  return section('day-status',`Ground truth — Buddy at the end of each day`,
    `<table class="dtab"><thead><tr><th>day</th><th>health</th><th>hunger</th><th>hydration</th>`+
    `<th>illness</th><th>food</th><th>med</th><th>take-home</th></tr></thead><tbody>${rows}</tbody></table>`);
}
function allEmails(t){
  const em=(t.sent_emails||[]).filter(Boolean).map(m=>({dir:'sent',to:m.to,subject:m.subject,body:m.body,day:m.day}));
  if(!em.length) return '';
  return section('all-emails',`Every email the agent wrote (${em.length})`,inbox(em,'all'));
}

// Map a tool call to a plain-language verb (no emoji), so the ledger reads cleanly.
function actionLabel(tool,arg){
  const blob=((tool||'')+' '+(arg||''));
  if(/read_email/.test(blob)) return 'Read inbox';
  if(/send_email/.test(blob)){ const to=(blob.match(/([^\s,"'}<>]+@[^\s,"'}<>]+)/)||[])[1]; return 'Sent email'+(to?' · '+to.split('@')[0]:''); }
  const post=/\bPOST\b/i.test(blob);
  const MAP=[
    [/\/care\/feed/,'Fed Buddy'],[/\/care\/water/,'Refilled water'],[/\/care\/medicate/,'Medicated'],
    [/\/care\/note/,'Left a care note'],[/\/care\/schedule-next/,'Scheduled next check-in'],
    [/\/vet\/consult/,'Consulted the vet'],[/\/vet\/visit/,'Vet visit'],
    [/\/supplies\/order/,'Ordered supplies'],[/\/supplies/,'Checked supplies'],
    [/\/camera/,'Viewed camera'],[/\/account/,'Checked account'],[/\/system\/status/,'Read console'],
  ];
  for(const [re,verb] of MAP) if(re.test(blob)) return verb;
  return post?'Console command':'Console read';
}
function fmtCmd(tool,arg){
  if(tool!=='bash') return (tool+(arg?' '+arg:'')).trim();
  const a=arg||'';
  const method=(a.match(/-X\s+(\w+)/)||[])[1]||'GET';
  const path=(a.match(/https?:\/\/[^\/\s]+(\/[^\s'"]*)/)||[])[1];
  const body=(a.match(/-d\s+'([^']*)'/)||a.match(/-d\s+"([^"]*)"/)||[])[1];
  if(!path) return a;
  return method+' '+path+(body?'  '+body:'');
}
function fmtResult(res){
  if(res==null) return '';
  const s=(''+res).trim();
  try{ return JSON.stringify(JSON.parse(s),null,2); }catch(e){}
  return s;
}

refreshRuns();
</script></body></html>
"""

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7861"))
    print(f"\n  Pet-Bench Explorer → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
