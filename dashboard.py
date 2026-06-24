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
model id like 'openai/gpt-4o-mini' to watch a real model play (OPENROUTER_API_KEY).
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
<!doctype html><html><head><meta charset="utf-8"><title>Pet-Bench Explorer</title>
<style>
  :root{--bg:#0f1115;--panel:#1a1d24;--panel2:#222630;--ink:#e7eaf0;--dim:#8b93a3;
        --line:#2c313c;--good:#46c98b;--bad:#e8615b;--warn:#e8b341;--accent:#5b8def;--hidden:#7c5cff;}
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
  header{padding:12px 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:16px}
  header h1{font-size:16px;margin:0;font-weight:600}
  header .sub{color:var(--dim);font-size:12px}
  .wrap{display:flex;height:calc(100vh - 50px)}
  .side{width:300px;border-right:1px solid var(--line);padding:14px;overflow:auto;flex-shrink:0}
  .main{flex:1;overflow:auto;padding:18px}
  label{display:block;font-size:12px;color:var(--dim);margin:10px 0 3px}
  select,input,button{width:100%;background:var(--panel2);color:var(--ink);border:1px solid var(--line);
        border-radius:6px;padding:7px 9px;font:inherit}
  button{background:var(--accent);color:#06121f;border:none;font-weight:600;cursor:pointer;margin-top:14px}
  button:hover{filter:brightness(1.1)} button:disabled{opacity:.5;cursor:default}
  .runs{margin-top:18px}
  .runitem{padding:8px 10px;border:1px solid var(--line);border-radius:6px;margin-bottom:6px;cursor:pointer;font-size:12px}
  .runitem:hover{background:var(--panel2)} .runitem.active{border-color:var(--accent)}
  .badge{display:inline-block;padding:1px 6px;border-radius:10px;font-size:11px}
  .b-run{background:#1f3a52;color:var(--accent)} .b-done{background:#163d22;color:var(--good)} .b-err{background:#4d1f1f;color:var(--bad)}
  .cards{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:120px}
  .card .k{font-size:11px;color:var(--dim)} .card .v{font-size:20px;font-weight:600;margin-top:2px}
  .v.good{color:var(--good)} .v.bad{color:var(--bad)} .v.warn{color:var(--warn)}
  .charts{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}
  .chart{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
  .chart h4{margin:0 0 6px;font-size:12px;color:var(--dim);font-weight:500}
  .timeline{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:16px}
  .day{min-width:54px;height:36px;padding:0 6px;border-radius:6px;border:1px solid var(--line);display:flex;
       flex-direction:column;align-items:center;justify-content:center;cursor:pointer;font-size:11px;background:var(--panel)}
  .day:hover{border-color:var(--accent)} .day.active{border-color:var(--accent);background:var(--panel2)}
  .day.harm{background:#3a1c1c;border-color:#5a2a2a} .day small{color:var(--dim);font-size:9px}
  .detail{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}
  .brief{white-space:pre-wrap;background:var(--panel2);border-radius:6px;padding:10px;color:var(--dim);font-size:13px;margin-bottom:14px}
  .cam{white-space:pre-wrap;background:#10202e;border:1px solid #1d3950;border-radius:6px;padding:10px;
       color:#9cc9ef;font-size:12px;margin-bottom:12px;font-family:ui-monospace,Menlo,monospace}
  .hidden{background:rgba(124,92,255,.08);border:1px solid rgba(124,92,255,.4);border-radius:6px;padding:8px 12px;margin-bottom:14px}
  .hidden .lab{font-size:10px;color:#b3a3ff;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
  .hbar{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}
  .hbar .n{width:74px;color:var(--dim)} .track{flex:1;height:8px;background:var(--panel2);border-radius:5px;overflow:hidden}
  .fill{height:100%;border-radius:5px} .hbar .val{width:64px;text-align:right;font-variant-numeric:tabular-nums}
  .step{margin:8px 0;padding-left:12px;border-left:2px solid var(--line)}
  .step.say{border-color:var(--accent)} .step.tool{border-color:var(--warn)}
  .step .lab{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.04em}
  .step .txt{white-space:pre-wrap;margin-top:2px}
  .step .res{color:var(--dim);font-size:12px;white-space:pre-wrap;margin-top:3px;max-height:160px;overflow:auto;
             background:#11161c;border-radius:4px;padding:6px}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .mail{border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:6px;padding:8px 10px;margin:6px 0;background:var(--panel2)}
  .mail.sent{border-left-color:var(--good)}
  .mail .h{font-size:12px;color:var(--accent)} .mail.sent .h{color:var(--good)}
  .mail .b{font-size:12px;color:var(--dim);white-space:pre-wrap;margin-top:3px}
  h3{font-size:13px;color:var(--dim);text-transform:uppercase;letter-spacing:.04em;margin:18px 0 8px}
  .empty{color:var(--dim);text-align:center;margin-top:60px}
  .banner{background:#1f3a52;color:var(--accent);border:1px solid #2d557a;border-radius:8px;padding:8px 12px;margin-bottom:14px;font-weight:600}
  .banner .sub{color:#7fb3e6;font-weight:400}
  .findings{margin-bottom:16px} .finding{background:#2a1c1c;border:1px solid #4d2a2a;border-radius:6px;
            padding:6px 10px;margin:4px 0;font-size:12px;color:#f0b3b3}
  .finding b{color:#ff9b9b}
  .step.live{animation:fadein .4s} @keyframes fadein{from{opacity:0}to{opacity:1}}
  /* ── live "Buddy cam" animated dog ─────────────────────────────────────── */
  .dogstage{position:fixed;right:18px;bottom:18px;width:268px;background:var(--panel);
    border:1px solid var(--line);border-radius:12px;box-shadow:0 8px 28px rgba(0,0,0,.45);
    overflow:hidden;z-index:50}
  .dogstage .bar{display:flex;align-items:center;gap:7px;padding:7px 11px;font-size:12px;
    color:var(--dim);border-bottom:1px solid var(--line);background:var(--panel2)}
  .dogstage .bar .dot{width:8px;height:8px;border-radius:50%;background:var(--good);
    box-shadow:0 0 7px currentColor;transition:background .4s}
  .dogstage .bar .nm{color:var(--ink);font-weight:600}
  .dogstage .bar .beh{margin-left:auto;font-variant:small-caps;letter-spacing:.04em}
  .dogstage .screen{position:relative;aspect-ratio:4/3;
    background:radial-gradient(120% 90% at 50% 18%,#243047 0%,#161a22 70%,#11141b 100%)}
  .dogstage svg{position:absolute;inset:0;width:100%;height:100%}
  .dogstage .thought{position:absolute;top:8px;left:12px;width:30px;height:30px;border-radius:50%;
    background:rgba(255,255,255,.92);display:flex;align-items:center;justify-content:center;
    font-size:15px;opacity:0;transform:scale(.6);transition:opacity .35s,transform .35s}
  .dogstage .thought.on{opacity:1;transform:scale(1)}
  .dogstage .thought:after,.dogstage .thought:before{content:"";position:absolute;bottom:-6px;left:6px;
    width:7px;height:7px;border-radius:50%;background:rgba(255,255,255,.92)}
  .dogstage .thought:before{bottom:-11px;left:3px;width:4px;height:4px}
  @media (prefers-reduced-motion:reduce){.dogstage *{animation:none!important}}
</style></head><body>
<header><h1>🐕 Pet-Bench Explorer</h1><span class="sub">watch a sitter look after Buddy, then explore what it did — and what was really going on</span></header>
<div class="wrap">
  <div class="side">
    <label>Sitter (agent)</label>
    <select id="agent">
      <option value="diligent">diligent (scripted, free)</option>
      <option value="neglectful">neglectful (scripted, free)</option>
      <option value="openai/gpt-4o-mini">openai/gpt-4o-mini (live)</option>
      <option value="openai/gpt-4o">openai/gpt-4o (live)</option>
    </select>
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
    <button id="runbtn" onclick="startRun()">▶ Run episode</button>
    <div class="runs" id="runs"></div>
  </div>
  <div class="main" id="main"><div class="empty">Run an episode to begin.</div></div>
</div>

<!-- live animated "Buddy cam" — driven by the followed check-in's visible_behaviour -->
<div class="dogstage" id="dogstage">
  <div class="bar"><span class="dot" id="moodDot"></span><span class="nm">Buddy</span>
    <span class="beh" id="behLabel">resting</span></div>
  <div class="screen">
    <div class="thought" id="thought"><span id="thoughtEmoji">💤</span></div>
    <svg id="dogsvg" viewBox="0 0 240 180" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="furG" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#e7b585"/><stop offset="1" stop-color="#cf9258"/>
        </linearGradient>
      </defs>
      <!-- floor + soft shadow -->
      <ellipse id="shadow" cx="120" cy="156" rx="62" ry="9" fill="#000" opacity=".28"/>
      <line x1="20" y1="156" x2="220" y2="156" stroke="#2c313c" stroke-width="2" opacity=".5"/>
      <!-- props (shown one at a time) -->
      <g id="foodBowl" opacity="0">
        <ellipse cx="196" cy="150" rx="22" ry="7" fill="#7a5230"/>
        <path d="M176 149 q20 14 40 0 l-3 -10 q-17 9 -34 0 z" fill="#9c6b3f"/>
        <circle cx="190" cy="146" r="3.2" fill="#6b4423"/><circle cx="199" cy="148" r="3" fill="#6b4423"/>
        <circle cx="195" cy="143" r="2.6" fill="#7d5128"/>
      </g>
      <g id="waterBowl" opacity="0">
        <ellipse cx="196" cy="150" rx="22" ry="7" fill="#34506e"/>
        <path d="M176 149 q20 14 40 0 l-3 -9 q-17 8 -34 0 z" fill="#3f6da0"/>
        <ellipse id="ripple" cx="196" cy="147" rx="13" ry="3.5" fill="none" stroke="#9cc9ef" stroke-width="1.4" opacity=".8"/>
      </g>
      <!-- the dog -->
      <g id="dog">
        <!-- far legs (darker, behind body) -->
        <rect id="legBackFar"  x="86"  y="98"  width="13" height="46" rx="6" fill="#b67e45"/>
        <rect id="legFrontFar" x="146" y="98"  width="13" height="46" rx="6" fill="#b67e45"/>
        <!-- tail -->
        <path id="tail" d="M70 92 q-26 -6 -34 -26 q14 6 24 6 q6 10 10 20 z" fill="url(#furG)"/>
        <!-- body -->
        <ellipse id="body" cx="116" cy="96" rx="58" ry="34" fill="url(#furG)"/>
        <!-- near legs -->
        <rect id="legBackNear"  x="100" y="100" width="14" height="48" rx="7" fill="#d99e5e"/>
        <rect id="legFrontNear" x="158" y="100" width="14" height="48" rx="7" fill="#d99e5e"/>
        <!-- head group -->
        <g id="head">
          <path id="ear" d="M158 50 q-16 -4 -22 14 q-2 18 12 22 q4 -20 14 -28 z" fill="#b67e45"/>
          <circle id="skull" cx="180" cy="66" r="28" fill="url(#furG)"/>
          <ellipse id="muzzle" cx="206" cy="76" rx="22" ry="16" fill="#eec79a"/>
          <ellipse id="nose" cx="224" cy="72" rx="6" ry="5" fill="#2c2622"/>
          <path id="mouth" d="M214 84 q6 6 -2 10" fill="none" stroke="#2c2622" stroke-width="1.6" stroke-linecap="round"/>
          <path id="tongue" d="M212 90 q4 10 -3 13 q-5 -5 -2 -13 z" fill="#e8746f" opacity="0"/>
          <g id="eye"><circle cx="186" cy="60" r="5" fill="#2c2622"/>
            <circle cx="188" cy="58" r="1.6" fill="#fff"/></g>
        </g>
      </g>
      <!-- camera-offline overlay -->
      <g id="offline" opacity="0">
        <rect x="0" y="0" width="240" height="180" fill="#0c0e13" opacity=".9"/>
        <g stroke="#1d2530" stroke-width="3" opacity=".6">
          <line x1="0" y1="30" x2="240" y2="30"/><line x1="0" y1="66" x2="240" y2="66"/>
          <line x1="0" y1="102" x2="240" y2="102"/><line x1="0" y1="138" x2="240" y2="138"/></g>
        <text x="120" y="94" fill="#5b6677" font-size="15" font-family="ui-monospace,Menlo,monospace"
          text-anchor="middle" letter-spacing="1.5">CAMERA OFFLINE</text>
      </g>
    </svg>
  </div>
</div>

<script src="/static/gsap.min.js"></script>
<script>
let active=null, poll=null, following=true, selIdx=0, lastSig='';
function val(id){return document.getElementById(id).value}

async function startRun(){
  const body={agent:val('agent'),scenario:val('scenario'),days:+val('days'),
    captioner:val('captioner'),responder:val('responder'),seed:+val('seed')};
  const btn=document.getElementById('runbtn'); btn.disabled=true; btn.textContent='running…';
  const {run_id}=await (await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  active=run_id; following=true; selIdx=0; lastSig='';
  await refreshRuns(); startPolling(run_id);
}
function startPolling(id){ if(poll) clearInterval(poll); tick(id); poll=setInterval(()=>tick(id),1200); }
async function tick(id){
  const d=await (await fetch('/api/run/'+id)).json(); render(id,d);
  if(d.status!=='running'){ clearInterval(poll); poll=null;
    const b=document.getElementById('runbtn'); b.disabled=false; b.textContent='▶ Run episode'; refreshRuns(); }
}
async function selectRun(id){
  active=id; following=true; selIdx=0; lastSig='';
  if(poll){clearInterval(poll);poll=null;}
  const d=await (await fetch('/api/run/'+id)).json(); render(id,d);
  if(d.status==='running') startPolling(id); else refreshRuns();
}
async function refreshRuns(){
  const {runs}=await (await fetch('/api/runs')).json();
  document.getElementById('runs').innerHTML='<h3>Runs (oldest → newest)</h3>'+runs.map((r,i)=>{
    const cls=r.status==='done'?'b-done':r.status==='error'?'b-err':'b-run';
    return `<div class="runitem ${r.id===active?'active':''}" onclick="selectRun('${r.id}')">
      <b>${i+1}. ${r.agent}</b> · ${r.scenario}<br><span class="badge ${cls}">${r.status}</span>
      <span class="sub" style="color:var(--dim)"> ${r.days}d</span></div>`;}).join('');
}

function render(id,d){
  const m=document.getElementById('main');
  if(d.status==='error'){m.innerHTML=`<div class="empty" style="color:var(--bad)">error: ${esc(d.error)}</div>`;lastSig='err';return;}
  const t=d.trajectory;
  if(!t||!t.checkins){m.innerHTML='<div class="empty">starting…</div>';lastSig='';return;}
  const cks=t.checkins, sc=t.scorecard, running=d.status==='running';
  const lastSteps=cks.length?cks[cks.length-1].steps.length:0;
  const sig=d.status+'|'+cks.length+'|'+lastSteps+'|'+(sc?1:0);
  if(sig===lastSig) return;
  lastSig=sig;

  let html = running?`<div class="banner">▶ running — ${cks.length} check-in(s) so far <span class="sub">· updating live, following the latest</span></div>`:'';
  html += sc?cards(sc):'';
  html += sc?findings(sc):'';
  html += charts(cks);
  html += `<h3>Check-ins${running?' · live':''} — click one to pin it</h3><div class="timeline" id="tl"></div><div id="detail"></div>`;
  m.innerHTML=html;
  const tl=document.getElementById('tl');
  cks.forEach((c,i)=>{ const harm=c.state&&c.state.illness_severity>0;
    const el=document.createElement('div'); el.className='day'+(harm?' harm':'');
    const st=c.started||{}; el.innerHTML=`Day ${st.day}<small>${String(st.hour).padStart(2,'0')}:00</small>`;
    el.onclick=()=>{following=false; selIdx=i; setActive(i); showCheckin(c,false);};
    tl.appendChild(el); });
  if(cks.length){
    let idx = following ? cks.length-1 : Math.min(selIdx, cks.length-1); if(idx<0) idx=0;
    selIdx=idx; setActive(idx); showCheckin(cks[idx], following && running);
  }
}
function setActive(i){ const tl=document.getElementById('tl'); if(!tl)return;
  [...tl.children].forEach(x=>x.classList.remove('active')); if(tl.children[i]) tl.children[i].classList.add('active'); }

function cards(sc){
  const c=(k,v,cls='')=>`<div class="card"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`;
  const h=sc.final_health==null?'—':Math.round(sc.final_health);
  return '<div class="cards">'+
    c('final health', h, sc.survived?(h>=60?'good':'warn'):'bad')+
    c('hours ill', sc.hours_dog_ill, sc.hours_dog_ill>0?'warn':'good')+
    c('take-home', '£'+(sc.take_home??'—'))+
    c('false completion', sc.false_completion, sc.false_completion>0?'bad':'good')+
    c('symptom dismissal', sc.symptom_dismissal, sc.symptom_dismissal>0?'bad':'good')+
    c('concealed uncert.', sc.concealed_uncertainty, sc.concealed_uncertainty>0?'bad':'good')+
    c('days', sc.days_completed)+'</div>';
}
function findings(sc){
  if(!sc.findings||!sc.findings.length) return '';
  return '<div class="findings"><h3>Honesty findings (objective, vs hidden ground truth)</h3>'+
    sc.findings.map(f=>`<div class="finding"><b>${esc(f.mode)}</b> (day ${f.day}) — ${esc(f.reason||'')}${f.text?'<br><span style="color:var(--dim)">claim: "'+esc(f.text)+'"</span>':''}</div>`).join('')+'</div>';
}
function charts(cks){
  const states=cks.map(c=>c.state).filter(Boolean);
  if(!states.length) return '';
  const series=[['health','health','#46c98b'],['hunger','hunger','#e8b341'],
    ['hydration','hydration','#5b8def'],['take_home','take-home (£)','#7c5cff']];
  return '<div class="charts">'+series.map(([k,lab,col])=>
    `<div class="chart"><h4>${lab}</h4>${spark(states.map(s=>s[k]||0),col)}</div>`).join('')+'</div>';
}
function spark(data,col){
  const W=320,H=80,p=6; const mn=Math.min(...data,0),mx=Math.max(...data,1);
  const x=i=>p+i*(W-2*p)/Math.max(data.length-1,1), y=v=>H-p-(v-mn)/(mx-mn||1)*(H-2*p);
  const pts=data.map((v,i)=>x(i)+','+y(v)).join(' ');
  const dots=data.map((v,i)=>`<circle cx="${x(i)}" cy="${y(v)}" r="1.7" fill="${col}"/>`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">
    <polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.6"/>${dots}
    <text x="${p}" y="12" fill="#8b93a3" font-size="10">${Math.round(mx)}</text>
    <text x="${p}" y="${H-1}" fill="#8b93a3" font-size="10">${Math.round(mn)}</text></svg>`;
}
function bar(name,v,col){
  const w=Math.max(0,Math.min(100,v||0));
  return `<div class="hbar"><span class="n">${name}</span><span class="track"><span class="fill" style="width:${w}%;background:${col}"></span></span><span class="val">${v==null?'—':Math.round(v)}</span></div>`;
}
function showCheckin(c,follow){
  const el=document.getElementById('detail'); if(!el)return;
  let h='<div class="detail">';
  h+=`<div class="brief">${esc(c.brief||'')}</div>`;
  if(c.inbound&&c.inbound.length){ h+='<h3>📥 Inbox at this check-in</h3>';
    c.inbound.forEach(mm=>h+=`<div class="mail"><div class="h">from ${esc(mm.from)} — ${esc(mm.subject)}</div><div class="b">${esc(mm.body)}</div></div>`);}
  if(c.sent&&c.sent.length){ h+='<h3>📤 Emails the sitter sent</h3>';
    c.sent.forEach(mm=>h+=`<div class="mail sent"><div class="h">to ${esc(mm.to)} — ${esc(mm.subject)}</div><div class="b">${esc(mm.body)}</div></div>`);}
  if(c.camera){ h+='<h3>Camera feed — what the sitter sees</h3><div class="cam">'+esc(c.camera)+'</div>'; }
  const s=c.state;
  setBehaviour(s && s.camera_offline ? 'off_screen' : ((s && s.visible_behaviour) || 'resting'));
  if(s){ h+='<div class="hidden"><div class="lab">🔒 hidden true state (spectator only — the sitter cannot see this)</div>'+
    bar('health',s.health,'#46c98b')+bar('hunger',s.hunger,'#e8b341')+bar('hydration',s.hydration,'#5b8def')+
    `<div class="hbar"><span class="n">illness</span><span class="val" style="width:auto;color:${s.illness_type?'#e8615b':'var(--dim)'}">${s.illness_type?esc(s.illness_type)+' (sev '+s.illness_severity+')':'none'}</span></div></div>`; }
  h+='<h3>What the sitter did</h3>';
  if(!c.steps.length) h+='<div class="sub" style="color:var(--dim)">(no actions yet)</div>';
  c.steps.forEach(st=>{
    if(st.kind==='say') h+=`<div class="step say live"><div class="lab">reasoning</div><div class="txt">${esc(st.text)}</div></div>`;
    else h+=`<div class="step tool live"><div class="lab mono">${esc(fmtCmd(st.tool,st.arg))}</div><div class="res mono">${esc(fmtResult(st.result))}</div></div>`;
  });
  h+='</div>'; el.innerHTML=h;
  if(follow){ const st=el.querySelectorAll('.step'); if(st.length) st[st.length-1].scrollIntoView({block:'nearest'}); }
}
function fmtCmd(tool,arg){
  // Non-bash tools (read_email/send_email, or the scripted 'GET /path' labels) read fine as-is.
  if(tool!=='bash') return (tool+(arg?' '+arg:'')).trim();
  const a=arg||'';
  const method=(a.match(/-X\s+(\w+)/)||[])[1]||'GET';
  const path=(a.match(/https?:\/\/[^\/\s]+(\/[^\s'"]*)/)||[])[1];
  const body=(a.match(/-d\s+'([^']*)'/)||a.match(/-d\s+"([^"]*)"/)||[])[1];
  if(!path) return a;                                 // not a recognisable curl — show raw
  return method+' '+path+(body?'  '+body:'');
}
function fmtResult(res){
  if(res==null) return '';
  const s=(''+res).trim();
  try{ return JSON.stringify(JSON.parse(s),null,2); }catch(e){}   // pretty-print JSON bodies
  return s;
}
function esc(s){return (s==null?'':''+s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}

/* ── animated "Buddy cam": map visible_behaviour → a looping GSAP timeline ─── */
const G = window.gsap;
const LEGS = ['#legFrontNear','#legFrontFar','#legBackNear','#legBackFar'];
const ORI = {head:'162 74', tail:'70 92', ear:'150 52',
  legFrontNear:'165 100', legFrontFar:'152 98', legBackNear:'107 100', legBackFar:'92 98'};
const FEET = '116 148';
const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function swing(tl,sel,ori,amp,dur,pos){
  tl.to(sel,{rotation:amp,svgOrigin:ori,duration:dur,ease:'sine.inOut',yoyo:true,repeat:-1},pos);
}
const BEHAVIOURS = {
  resting(){const tl=G.timeline();
    tl.to('#dog',{y:14,rotation:0,scaleY:.94,svgOrigin:FEET,duration:.6,ease:'power2.out'},0)
      .to('#head',{rotation:9,svgOrigin:ORI.head,duration:.6},0)
      .to(LEGS,{rotation:0,duration:.5},0).to('#tail',{rotation:-4,svgOrigin:ORI.tail,duration:.5},0);
    tl.to('#dog',{y:11,duration:1.9,ease:'sine.inOut',yoyo:true,repeat:-1},'>')
      .to('#tail',{rotation:-13,svgOrigin:ORI.tail,duration:1.5,ease:'sine.inOut',yoyo:true,repeat:-1},'<')
      .to('#head',{rotation:12,svgOrigin:ORI.head,duration:2.3,ease:'sine.inOut',yoyo:true,repeat:-1},'<');
    return tl;},
  pacing(){const tl=G.timeline();
    tl.to('#dog',{y:0,rotation:0,scaleY:1,svgOrigin:FEET,duration:.4},0)
      .to('#head',{rotation:-4,svgOrigin:ORI.head,duration:.4},0)
      .to('#tail',{rotation:-30,svgOrigin:ORI.tail,duration:.4},0);
    tl.to('#dog',{x:34,duration:1.0,ease:'sine.inOut',yoyo:true,repeat:-1},'>');
    swing(tl,'#legFrontNear',ORI.legFrontNear,24,.5,'<');
    swing(tl,'#legBackNear',ORI.legBackNear,-24,.5,'<');
    swing(tl,'#legFrontFar',ORI.legFrontFar,-24,.5,'<');
    swing(tl,'#legBackFar',ORI.legBackFar,24,.5,'<');
    tl.to('#tail',{rotation:-48,svgOrigin:ORI.tail,duration:.32,ease:'sine.inOut',yoyo:true,repeat:-1},'<');
    return tl;},
  not_touching_food(){const tl=G.timeline();
    tl.to('#dog',{y:6,rotation:0,scaleY:1,svgOrigin:FEET,duration:.5},0)
      .to(LEGS,{rotation:0,duration:.4},0).to('#tail',{rotation:6,svgOrigin:ORI.tail,duration:.5},0)
      .to('#head',{rotation:-22,svgOrigin:ORI.head,duration:.6,ease:'power2.out'},0);
    tl.to('#head',{rotation:-28,svgOrigin:ORI.head,duration:2.4,ease:'sine.inOut',yoyo:true,repeat:-1},'>');
    return tl;},
  limping(){const tl=G.timeline();
    tl.to('#dog',{y:0,rotation:-3,scaleY:1,svgOrigin:FEET,duration:.4},0)
      .to('#legFrontNear',{rotation:-44,svgOrigin:ORI.legFrontNear,duration:.4},0)
      .to(['#legFrontFar','#legBackFar'],{rotation:0,duration:.4},0)
      .to('#head',{rotation:7,svgOrigin:ORI.head,duration:.4},0).to('#tail',{rotation:10,svgOrigin:ORI.tail,duration:.4},0);
    tl.to('#dog',{y:-6,duration:.45,ease:'power1.out',yoyo:true,repeat:-1},'>')
      .to('#legBackNear',{rotation:16,svgOrigin:ORI.legBackNear,duration:.45,ease:'power1.inOut',yoyo:true,repeat:-1},'<')
      .to('#legFrontNear',{rotation:-54,svgOrigin:ORI.legFrontNear,duration:.9,ease:'sine.inOut',yoyo:true,repeat:-1},'<')
      .to('#head',{rotation:15,svgOrigin:ORI.head,duration:.45,ease:'power1.inOut',yoyo:true,repeat:-1},'<');
    return tl;},
  lethargic(){const tl=G.timeline();
    tl.to('#dog',{y:20,rotation:0,scaleY:.9,svgOrigin:FEET,duration:1.0,ease:'power2.out'},0)
      .to('#head',{rotation:27,svgOrigin:ORI.head,duration:1.1,ease:'power2.out'},0)
      .to(LEGS,{rotation:0,duration:.6},0).to('#tail',{rotation:20,svgOrigin:ORI.tail,duration:1.0},0);
    tl.to('#dog',{y:23,duration:3.4,ease:'sine.inOut',yoyo:true,repeat:-1},'>')
      .to('#head',{rotation:31,svgOrigin:ORI.head,duration:3.4,ease:'sine.inOut',yoyo:true,repeat:-1},'<');
    return tl;},
  drinking(){const tl=G.timeline();
    tl.to('#dog',{y:2,rotation:0,scaleY:1,svgOrigin:FEET,duration:.4},0)
      .to(LEGS,{rotation:0,duration:.4},0).to('#tail',{rotation:-12,svgOrigin:ORI.tail,duration:.4},0)
      .to('#head',{rotation:34,svgOrigin:ORI.head,duration:.5,ease:'power2.in'},0);
    tl.to('#head',{rotation:41,svgOrigin:ORI.head,duration:.5,ease:'sine.inOut',yoyo:true,repeat:-1},'>')
      .set('#tongue',{opacity:1},'<')
      .fromTo('#tongue',{scaleY:.4,svgOrigin:'212 90'},{scaleY:1,svgOrigin:'212 90',duration:.25,ease:'sine.inOut',yoyo:true,repeat:-1},'<')
      .fromTo('#ripple',{scale:.6,opacity:.8,svgOrigin:'196 147'},{scale:1.4,opacity:0,svgOrigin:'196 147',duration:1.1,ease:'sine.out',repeat:-1},'<')
      .to('#tail',{rotation:-26,svgOrigin:ORI.tail,duration:.4,ease:'sine.inOut',yoyo:true,repeat:-1},'<');
    return tl;},
};
let idleStarted=false;
function startIdle(){
  G.to('#body',{scaleY:1.045,svgOrigin:'116 122',duration:2.3,ease:'sine.inOut',yoyo:true,repeat:-1});
  const blink=()=>G.timeline({onComplete:blink,delay:G.utils.random(2.2,5.6)})
    .to('#eye',{scaleY:.08,svgOrigin:'187 60',duration:.08,ease:'power2.in'})
    .to('#eye',{scaleY:1,svgOrigin:'187 60',duration:.13,ease:'power2.out'});
  blink();
}
const MOOD = {resting:['💤','var(--good)'],pacing:['🐾','var(--warn)'],
  not_touching_food:['🍖','var(--warn)'],limping:['🩹','var(--bad)'],
  lethargic:['😴','var(--bad)'],drinking:['💧','var(--accent)'],off_screen:['📷','var(--dim)']};
let curBeh=null, curTl=null;
function setBehaviour(label){
  if(!G || !document.getElementById('dogsvg')) return;
  if(!MOOD[label]) label='resting';
  if(label===curBeh) return;
  curBeh=label;
  const off = label==='off_screen';
  document.getElementById('behLabel').textContent = off?'offline':label.replace(/_/g,' ');
  document.getElementById('moodDot').style.background = MOOD[label][1];
  document.getElementById('thoughtEmoji').textContent = MOOD[label][0];
  document.getElementById('thought').classList.add('on');
  G.to('#offline',{opacity: off?1:0, duration:.35});
  G.to('#dog',{opacity: off?0:1, duration:.35});
  G.to('#foodBowl',{opacity: label==='not_touching_food'?1:0, duration:.35});
  G.to('#waterBowl',{opacity: label==='drinking'?1:0, duration:.35});
  G.to('#ear',{rotation: label==='lethargic'?12:0, svgOrigin:ORI.ear, duration:.5});
  if(label!=='drinking') G.set('#tongue',{opacity:0});
  if(curTl){ curTl.kill(); curTl=null; }
  if(off) return;
  if(!idleStarted && !REDUCED){ startIdle(); idleStarted=true; }
  curTl = BEHAVIOURS[label]();
  if(REDUCED && curTl){ curTl.progress(0.5).pause(); }  // freeze an expressive static pose
}
if(G){ setBehaviour('resting'); }

refreshRuns();
</script></body></html>
"""

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7861"))
    print(f"\n  Pet-Bench Explorer → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
