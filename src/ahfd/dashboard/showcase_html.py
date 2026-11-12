"""The showcase page (/showcase): the live system explained on one screen.

For demonstrations, not the ward. Camera and depth side by side, each bed's
progress through the seven bed-exit phases, live head and hip heights, the
pipeline with its live values, and every alert as it happens. It reads only
/api/state and the two MJPEG streams, so it adds no load beyond one more
viewer. One self-contained string with no outside assets (works offline).
"""

from __future__ import annotations

SHOWCASE_HTML = r"""<!doctype html>
<html lang="en" data-level="ok">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Live showcase - bed-exit and fall monitor</title>
<style>
  :root { color-scheme:dark;
    --bg:#0b1315; --panel:#121d20; --panel2:#17262a; --ink:#e9efec; --soft:#9fb1ae; --rule:#26383c;
    --sign:#0e3b3d; --teal:#2fb6a3; --red:#d8352a; --amber:#f2a30f; --go:#6fd3a6; --blue:#5aa9ff;
    --mattress:#33464a; --frame:#7d9094; --blanket:#3f6a65; --pillow:#57696c; --figure:#e9efec; }
  * { box-sizing:border-box; }
  [hidden] { display:none !important; }
  html, body { margin:0; background:var(--bg); color:var(--ink); }
  body { font:16px/1.35 Bahnschrift,"DIN Alternate","Avenir Next","Roboto Condensed","Segoe UI",system-ui,sans-serif; }
  h1,h2,h3,p { margin:0; font-weight:inherit; font-size:inherit; }
  button { font:inherit; color:inherit; background:transparent; border:1px solid var(--soft); border-radius:6px;
           padding:6px 12px; cursor:pointer; min-height:36px; }
  button[aria-pressed="true"] { background:var(--ink); color:var(--bg); border-color:var(--ink); font-weight:700; }
  a { color:inherit; }
  .mono { font-family:ui-monospace,"Cascadia Mono",Consolas,monospace; font-variant-numeric:tabular-nums; }

  header { display:flex; flex-wrap:wrap; align-items:center; gap:10px 22px; padding:12px 22px; background:var(--sign); }
  [data-level="urgent"] header { background:var(--red); }
  [data-level="check"] header { background:var(--amber); color:#1d1503; }
  [data-level="down"] header { background:#111; background-image:repeating-linear-gradient(135deg,#111 0 22px,#3a2a05 22px 44px); }
  header .title h1 { font-size:24px; font-weight:700; letter-spacing:-.01em; }
  header .title p { font-size:14px; opacity:.85; }
  header .status { font-size:22px; font-weight:700; padding:4px 14px; border-radius:999px; background:rgba(0,0,0,.25); }
  header .chips { display:flex; gap:8px; flex-wrap:wrap; margin-left:auto; align-items:center; }
  .chip { font-size:14px; padding:4px 10px; border-radius:999px; background:rgba(0,0,0,.25); white-space:nowrap; }
  .chip b { font-size:16px; }
  header .clock { font-size:34px; line-height:1; }

  #alerts .strip { display:flex; align-items:center; gap:18px; padding:10px 22px; animation:pulse 1.2s ease-in-out infinite; }
  #alerts .strip.urgent { background:var(--red); color:#fff; }
  #alerts .strip.check { background:var(--amber); color:#1d1503; animation:none; }
  #alerts .strip h2 { font-size:30px; font-weight:700; }
  #alerts .strip .where { font-size:18px; }
  #alerts .strip button { margin-left:auto; background:#fff; color:#18282b; border-color:#fff; font-weight:700; font-size:17px; padding:8px 22px; }
  @keyframes pulse { 50% { filter:brightness(1.18); } }
  @media (prefers-reduced-motion:reduce){ #alerts .strip { animation:none; } }

  main { display:grid; grid-template-columns:minmax(0,1fr) 360px; gap:16px; padding:16px 22px 22px; }
  @media (max-width:1250px){ main { grid-template-columns:minmax(0,1fr); } }
  .panel { background:var(--panel); border:1px solid var(--rule); border-radius:10px; padding:12px 14px; min-width:0; }
  .label { display:flex; align-items:center; justify-content:space-between; gap:10px; font-size:13px; font-weight:700;
           letter-spacing:.14em; text-transform:uppercase; color:var(--soft); margin-bottom:8px; }
  .label .tools { display:flex; gap:6px; letter-spacing:0; text-transform:none; font-weight:400; }
  .label .tools button { min-height:30px; padding:3px 10px; font-size:13px; }

  .feeds { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:900px){ .feeds { grid-template-columns:1fr; } }
  .feed { position:relative; background:#050708; border-radius:8px; overflow:hidden; aspect-ratio:16/9; }
  .feed img { width:100%; height:100%; object-fit:contain; display:block; }
  .feed .tag { position:absolute; left:8px; bottom:8px; padding:3px 9px; border-radius:6px; background:rgba(5,7,8,.8); font-size:13px; }
  .feed .note { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; text-align:center;
                padding:20px; color:var(--soft); font-size:15px; }
  .caption { font-size:14px; color:var(--soft); margin-top:8px; }
  .caption b { color:var(--ink); }

  .beds { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; margin-top:16px; }
  .bed { background:var(--panel); border:1px solid var(--rule); border-left:8px solid var(--rule); border-radius:10px; padding:12px 14px; }
  .bed.urgent { border-color:var(--red); }
  .bed.check { border-color:var(--amber); }
  .bed .top { display:grid; grid-template-columns:150px minmax(0,1fr); gap:12px; align-items:center; }
  .bed .art svg { display:block; width:100%; height:auto; }
  .bed .name { font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:var(--soft); font-weight:700; }
  .bed .now { font-size:30px; font-weight:700; line-height:1.05; }
  .bed.urgent .now { color:#ff8b7f; }
  .bed.check .now { color:#ffc660; }
  .bed .since { font-size:14px; color:var(--soft); }
  .bed .levels { display:flex; gap:0; margin-top:8px; }
  .bed .levels button { border-radius:0; margin-left:-1px; min-height:32px; padding:3px 12px; font-size:14px; }
  .bed .levels button:first-child { border-radius:6px 0 0 6px; }
  .bed .levels button:last-child { border-radius:0 6px 6px 0; }
  .phases { display:grid; grid-template-columns:repeat(7,1fr); gap:3px; margin:12px 0 4px; }
  .phases div { position:relative; padding:6px 4px 5px; border-radius:4px; background:var(--panel2); text-align:center;
                font-size:11.5px; line-height:1.15; color:var(--soft); min-height:44px; display:flex; align-items:center; justify-content:center; }
  .phases div.past { background:#1d3236; color:var(--ink); }
  .phases div.on { background:var(--teal); color:#062220; font-weight:700; }
  .phases div.on.warn { background:var(--amber); color:#1d1503; }
  .phases div.on.out { background:var(--red); color:#fff; }
  .phases div.model::after { content:"model"; position:absolute; top:-9px; left:50%; transform:translateX(-50%);
                font-size:10px; padding:0 5px; border-radius:6px; background:var(--blue); color:#06121f; font-weight:700; }
  .legend-row { display:flex; justify-content:space-between; gap:10px; font-size:12px; color:var(--soft); }
  .chartbox { margin-top:8px; }
  canvas.chart { width:100%; height:130px; display:block; background:#0d171a; border-radius:6px; }
  .keys { display:flex; gap:14px; font-size:12px; color:var(--soft); margin-top:4px; }
  .keys i { display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:5px; }
  .nobeds { padding:20px; text-align:center; color:var(--soft); border:2px dashed var(--rule); border-radius:10px; margin-top:16px; }

  aside { display:grid; gap:16px; align-content:start; }
  .steps { display:grid; gap:6px; }
  .step { display:grid; grid-template-columns:28px minmax(0,1fr); gap:10px; align-items:center; padding:7px 8px;
          border-radius:8px; background:var(--panel2); border:1px solid transparent; transition:border-color .3s, background .3s; }
  .step .n { width:26px; height:26px; border-radius:50%; display:flex; align-items:center; justify-content:center;
             background:var(--rule); font-weight:700; font-size:13px; }
  .step.live { border-color:var(--teal); }
  .step.live .n { background:var(--teal); color:#062220; }
  .step.alarm { border-color:var(--red); background:#3c1714; }
  .step.alarm .n { background:var(--red); color:#fff; }
  .step b { display:block; font-size:15px; }
  .step span { font-size:13px; color:var(--soft); }
  .step .v { color:var(--ink); }
  .events { max-height:34vh; overflow:auto; }
  .ev { display:grid; grid-template-columns:70px 10px minmax(0,1fr); gap:8px; align-items:baseline; padding:6px 0; border-bottom:1px solid var(--rule); font-size:14px; }
  .ev .dot { width:10px; height:10px; border-radius:50%; background:var(--soft); align-self:center; }
  .ev.urgent .dot { background:var(--red); }
  .ev.check .dot { background:var(--amber); }
  .ev .where { color:var(--soft); font-size:13px; }
  .tally { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; }
  .tally div { background:var(--panel2); border-radius:8px; padding:8px; }
  .tally b { display:block; font-size:28px; line-height:1; }
  .tally span { font-size:12px; color:var(--soft); }
  .privacy { font-size:14px; color:var(--soft); }
  .privacy b { color:var(--go); }
  .row-actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
</style>
</head>
<body>
<header>
  <div class="title"><h1>Bed-exit and fall monitor</h1><p id="sub">Live from the depth camera, processed on the edge computer</p></div>
  <div class="status" id="status" role="status">Starting</div>
  <div class="chips">
    <span class="chip"><b class="mono" id="c-fps">0</b> fps</span>
    <span class="chip"><b class="mono" id="c-people">0</b> in view</span>
    <span class="chip" id="c-depth">depth</span>
    <span class="chip" id="c-model" hidden>model</span>
    <button id="sound" class="chip">Sound off</button>
    <a class="chip" href="/ward">Nurse view</a>
    <a class="chip" href="/">Engineer view</a>
    <span class="clock mono" id="clock"></span>
  </div>
</header>
<section id="alerts" aria-live="assertive"></section>

<main>
  <div>
    <div class="feeds">
      <section class="panel">
        <div class="label">What the camera sees <span class="tools"><button id="rgb" aria-pressed="true">Camera image</button><button id="full1">Full screen</button></span></div>
        <div class="feed" id="feed1"><img id="stream" src="/stream.mjpg" alt="Live camera view with skeletons"/><div class="tag" id="tag1">Connecting</div></div>
        <p class="caption"><b>Pose model:</b> every person becomes 17 joints, tracked with their own ID. With the camera image off, only the stick figures remain: that is all the system keeps.</p>
      </section>
      <section class="panel">
        <div class="label">What depth measures <span class="tools">
          <button data-dmode="height" aria-pressed="true">Height above floor</button><button data-dmode="distance" aria-pressed="false">Distance</button><button id="full2">Full screen</button></span></div>
        <div class="feed" id="feed2"><img id="depth" alt="Live depth view"/><div class="note" id="depthnote" hidden></div><div class="tag" id="tag2">Depth</div></div>
        <p class="caption" id="depthcap"><b>Height map:</b> each pixel coloured by its height above the floor. Floor blue, mattress green, a standing head red. The system reasons in these metres, so the same rules work for any bed and any camera angle.</p>
      </section>
    </div>
    <div class="beds" id="beds"></div>
    <div class="nobeds" id="nobeds" hidden>No beds are set up for this camera. People are still tracked and falls still detected.</div>
  </div>

  <aside>
    <section class="panel">
      <div class="label">How it works, live</div>
      <div class="steps">
        <div class="step" id="s1"><div class="n">1</div><div><b>Depth camera</b><span>Colour and depth, <span class="v" id="v1">-</span></span></div></div>
        <div class="step" id="s2"><div class="n">2</div><div><b>Pose model (RTMO)</b><span><span class="v" id="v2">-</span></span></div></div>
        <div class="step" id="s3"><div class="n">3</div><div><b>Depth to height</b><span><span class="v" id="v3">-</span></span></div></div>
        <div class="step" id="s4"><div class="n">4</div><div><b>Bed-exit phases</b><span><span class="v" id="v4">-</span></span></div></div>
        <div class="step" id="s5"><div class="n">5</div><div><b>Fall check</b><span><span class="v" id="v5">-</span></span></div></div>
        <div class="step" id="s6"><div class="n">6</div><div><b>Alert to the nurse</b><span><span class="v" id="v6">-</span></span></div></div>
      </div>
      <p class="privacy" style="margin-top:10px"><b>No video is stored.</b> Images live only in memory and are dropped after pose; only joints and alerts are kept.</p>
    </section>
    <section class="panel">
      <div class="label">Since this page opened</div>
      <div class="tally">
        <div><b id="n-falls">0</b><span>Falls</span></div>
        <div><b id="n-possible">0</b><span>Possible falls</span></div>
        <div><b id="n-exit">0</b><span>Left bed</span></div>
        <div><b id="n-up">0</b><span>Getting up</span></div>
      </div>
    </section>
    <section class="panel">
      <div class="label">Event timeline <span class="tools"><button id="ackall">Acknowledge all</button></span></div>
      <div class="events" id="events"><div class="ev"><span></span><span></span><span class="where">Nothing yet. Try lying down, sitting up, then standing.</span></div></div>
    </section>
  </aside>
</main>

<script>
const ALERTS = {
  FALL_CONFIRMED:{rank:6,tier:'urgent',title:'Fall: still on the floor'},
  PERSON_DOWN:{rank:5,tier:'urgent',title:'Person on the floor'},
  FALL_SUSPECTED:{rank:4,tier:'urgent',title:'Fall detected'},
  POSSIBLE_FALL:{rank:3,tier:'check',title:'Possible fall'},
  NEAR_MISS:{rank:2,tier:'check',title:'Fell and got back up'},
  BED_EXIT:{rank:2,tier:'check',title:'Patient left the bed'},
  BED_EXIT_WARNING:{rank:1,tier:'check',title:'Patient may be getting up'},
};
const PHASES = [['RECLINED','Lying'],['TORSO_RISING','Sitting up'],['UPRIGHT_IN_BED','Upright in bed'],
  ['SHIFTING_TO_EDGE','Moving to edge'],['EDGE_SITTING','At the edge'],['ATTEMPTING_STAND','Trying to stand'],['OUT_OF_BED','Out of bed']];
const PIDX = Object.fromEntries(PHASES.map(([p],i) => [p,i]));
const STATE = {UPRIGHT:'Standing',SITTING:'Sitting',IN_BED:'In bed',FALLING:'Falling',ON_GROUND:'On the floor',
  RECOVERED:'Got back up',LOW_CONFIDENCE:'Cannot see clearly',UNKNOWN:'Cannot tell',TRACKED:'In view'};
const POSES = {
  none:'<rect x="150" y="72" width="62" height="13" rx="3" fill="var(--blanket)"/>',
  lying:'<circle cx="62" cy="61" r="17" fill="var(--figure)"/><path d="M78 70 L112 73" class="fig"/><path d="M92 84 C96 58 196 58 214 84 Z" fill="var(--blanket)"/>',
  situp:'<circle cx="70" cy="26" r="17" fill="var(--figure)"/><path d="M74 42 L92 78" class="fig"/><path d="M92 78 L200 76" class="fig"/><path d="M104 84 C108 62 196 62 214 84 Z" fill="var(--blanket)"/>',
  edge:'<circle cx="222" cy="24" r="17" fill="var(--figure)"/><path d="M222 40 L218 80 L250 82 L250 142 L266 144" class="fig"/><path d="M221 50 L240 74" class="fig thin"/>',
  stand:'<circle cx="270" cy="14" r="17" fill="var(--figure)"/><path d="M270 30 L270 88 M270 88 L258 148 M270 88 L284 148 M270 40 L254 76 M270 40 L288 74" class="fig"/>',
  falling:'<circle cx="318" cy="66" r="17" fill="var(--figure)"/><path d="M304 76 L262 112 M262 112 L232 146 M262 112 L286 148 M296 82 L306 116 M296 82 L336 96" class="fig"/>',
  floor:'<circle cx="238" cy="136" r="17" fill="var(--figure)"/><path d="M254 142 L314 145 L338 128 L352 146 M268 142 L292 124" class="fig"/>',
  unknown:'<text x="285" y="112" text-anchor="middle" font-size="60" font-weight="700" fill="var(--soft)">?</text>',
};
function drawing(pose){
  return `<svg viewBox="0 0 360 164" aria-hidden="true"><style>.fig{fill:none;stroke:var(--figure);stroke-width:14;stroke-linecap:round;stroke-linejoin:round}.thin{stroke-width:11}</style>
    <path d="M6 156 H354" stroke="var(--frame)" stroke-width="3"/><rect x="20" y="48" width="10" height="100" fill="var(--frame)"/>
    <rect x="30" y="106" width="190" height="9" fill="var(--frame)"/><path d="M52 115 V144 M198 115 V144" stroke="var(--frame)" stroke-width="7"/>
    <circle cx="52" cy="148" r="7" fill="var(--frame)"/><circle cx="198" cy="148" r="7" fill="var(--frame)"/>
    <rect x="30" y="84" width="190" height="22" rx="4" fill="var(--mattress)"/><rect x="34" y="72" width="50" height="14" rx="6" fill="var(--pillow)"/>
    ${POSES[pose] || ''}</svg>`;
}
const $ = id => document.getElementById(id);
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => '&#' + c.charCodeAt(0) + ';');
const info = t => ALERTS[t] || {rank:0, tier:'', title:String(t).replace(/_/g,' ').toLowerCase()};
function bedName(id){ const n = String(id).replace(/_/g,' ').replace(/\(.*?\)/g,'').trim(); const c = n.charAt(0).toUpperCase() + n.slice(1); return /bed/i.test(c) ? c : c + ' bed'; }
function setHtml(el, html){ if(el._h !== html){ el._h = html; el.innerHTML = html; } }
async function post(path, body){ try { const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body || {})}); return await r.json(); } catch(_){ return {ok:false}; } }

function reading(t){
  const ph = t.bed_phase, st = t.state;
  if(st === 'ON_GROUND') return {text:'On the floor', pose:'floor', tier:'urgent'};
  if(st === 'FALLING') return {text:'Falling', pose:'falling', tier:'urgent'};
  if(t.possible_fall) return {text:'Possible fall', pose:'falling', tier:'check'};
  if(ph === 'ATTEMPTING_STAND') return {text:'Trying to stand', pose:'stand', tier:'check'};
  if(ph === 'EDGE_SITTING' || ph === 'SHIFTING_TO_EDGE') return {text:'At the bed edge', pose:'edge', tier:'check'};
  if(ph === 'TORSO_RISING') return {text:'Sitting up', pose:'situp', tier:''};
  if(ph === 'OUT_OF_BED') return {text:'Out of bed', pose:'stand', tier:'check'};
  if(st === 'IN_BED' || ph === 'RECLINED') return {text:'Lying in bed', pose:'lying', tier:''};
  if(st === 'SITTING' || ph === 'UPRIGHT_IN_BED') return {text:'Sitting up in bed', pose:'situp', tier:''};
  if(st === 'UPRIGHT' || st === 'RECOVERED') return {text:STATE[st], pose:'stand', tier:''};
  return {text:STATE[st] || 'In view', pose:'unknown', tier:''};
}

// ---- live height history per bed (last 30 s) ----
const HIST_S = 30, hist = {}, since = {};
function remember(key, t){
  const now = Date.now() / 1000;
  const h = hist[key] = hist[key] || [];
  h.push({t:now, head:t ? t.head_h : null, hip:t ? (t.hip_h != null ? t.hip_h : t.height_m) : null});
  while(h.length && now - h[0].t > HIST_S) h.shift();
}
function drawChart(cv, key){
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if(cv.width !== w * dpr){ cv.width = w * dpr; cv.height = h * dpr; }
  const g = cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);
  const top = 2.0, L = 34, R = 6, T = 6, B = 16;
  const y = m => T + (1 - Math.min(top, Math.max(0, m)) / top) * (h - T - B);
  g.font = '11px system-ui'; g.fillStyle = '#9fb1ae'; g.strokeStyle = '#26383c'; g.lineWidth = 1;
  for(const m of [0, 0.5, 1.0, 1.5, 2.0]){ g.beginPath(); g.moveTo(L, y(m)); g.lineTo(w - R, y(m)); g.stroke(); g.fillText(m.toFixed(1) + ' m', 2, y(m) + 4); }
  g.fillText('floor', L + 4, y(0) - 3); g.fillText('now', w - R - 22, h - 3); g.fillText('-' + HIST_S + ' s', L, h - 3);
  const pts = hist[key] || [], now = Date.now() / 1000;
  const x = t => L + (1 - (now - t) / HIST_S) * (w - L - R);
  for(const [field, col] of [['hip', '#2fb6a3'], ['head', '#f2a30f']]){
    g.strokeStyle = col; g.lineWidth = 2.5; g.beginPath(); let on = false;
    for(const p of pts){ const v = p[field]; if(v == null){ on = false; continue; } const X = x(p.t), Y = y(v); if(!on){ g.moveTo(X, Y); on = true; } else g.lineTo(X, Y); }
    g.stroke();
    const last = [...pts].reverse().find(p => p[field] != null);
    if(last){ g.fillStyle = col; g.beginPath(); g.arc(x(last.t), y(last[field]), 4, 0, 7); g.fill();
      g.fillText(last[field].toFixed(2) + ' m', Math.min(w - 52, x(last.t) - 50), y(last[field]) - 7); }
  }
}

let bedKey = '';
function renderBeds(beds, tracks, open, watching){
  const box = $('beds');
  $('nobeds').hidden = beds.length > 0;
  const key = JSON.stringify(beds);
  if(key !== bedKey){
    bedKey = key;
    box.innerHTML = beds.map(b => `<article class="bed" data-bed="${esc(b.bed_id)}">
      <div class="top"><div class="art"></div><div>
        <div class="name">${esc(bedName(b.bed_id))}</div><div class="now">-</div><div class="since"></div>
        <div class="levels" role="group" aria-label="Alert level">${[['low','Low'],['medium','Medium'],['high','High']].map(([m,w]) =>
          `<button data-level="${m}" aria-pressed="${b.mode === m}" title="${m === 'low' ? 'Falls only' : m === 'medium' ? 'Also when the patient has left the bed' : 'Also when they start getting up'}">${w}</button>`).join('')}</div>
      </div></div>
      <div class="phases">${PHASES.map(([p, w]) => `<div data-p="${p}">${w}</div>`).join('')}</div>
      <div class="legend-row"><span>Bed-exit progress (rules); blue tag = learned model</span><span class="cusum"></span></div>
      <div class="chartbox"><canvas class="chart"></canvas>
        <div class="keys"><span><i style="background:#f2a30f"></i>head height</span><span><i style="background:#2fb6a3"></i>hip height</span></div></div>
    </article>`).join('');
    box.querySelectorAll('.bed').forEach(card => card.querySelectorAll('.levels button').forEach(btn =>
      btn.addEventListener('click', async () => { await post('/api/bed-mode', {bed_id:card.dataset.bed, mode:btn.dataset.level}); bedKey = ''; refresh(); })));
  }
  box.querySelectorAll('.bed').forEach(card => {
    const id = card.dataset.bed;
    const here = tracks.filter(t => t.bed_id === id);
    const t = here[0];
    remember(id, watching ? t : null);
    const r = t ? reading(t) : null;
    const alerts = open.filter(e => (e.zone || e.near_bed) === id).map(e => info(e.type)).sort((a,b) => b.rank - a.rank);
    const urgent = (r && r.tier === 'urgent') || alerts.some(a => a.tier === 'urgent');
    const check = !urgent && ((r && r.tier === 'check') || alerts.length > 0);
    card.className = 'bed' + (urgent ? ' urgent' : check ? ' check' : '');
    const text = !watching ? 'Not monitoring' : r ? r.text : alerts.length ? alerts[0].title : 'Empty bed';
    card.querySelector('.now').textContent = text;
    if(since[id] == null || since[id].text !== text) since[id] = {text, at:Date.now()};
    const secs = Math.floor((Date.now() - since[id].at) / 1000);
    card.querySelector('.since').textContent = secs < 3 ? '' : 'for ' + (secs < 60 ? secs + ' s' : Math.floor(secs / 60) + ' min ' + (secs % 60) + ' s');
    const pose = !watching ? 'none' : r ? r.pose : 'none';
    const art = card.querySelector('.art');
    if(art.dataset.pose !== pose){ art.dataset.pose = pose; art.innerHTML = drawing(pose); }
    const cur = t && t.bed_phase in PIDX ? PIDX[t.bed_phase] : -1;
    const model = t && t.model_phase in PIDX ? PIDX[t.model_phase] : -1;
    card.querySelectorAll('.phases div').forEach((el, i) => {
      el.className = (i < cur ? 'past' : '') + (i === cur ? ' on' + (i >= 3 && i <= 5 ? ' warn' : i === 6 ? ' out' : '') : '') + (i === model ? ' model' : '');
    });
    card.querySelector('.cusum').textContent = t && t.cusum_g != null ? 'movement score ' + t.cusum_g.toFixed(1) : '';
    drawChart(card.querySelector('canvas'), id);
  });
}

function strip(e){
  const a = info(e.type);
  return `<div class="strip ${a.tier}"><h2>${esc(a.title)}</h2><span class="where">${esc((e.zone || e.near_bed) ? bedName(e.zone || e.near_bed) : 'In the room')} &middot; ${esc(e.clock)}</span>
    <button onclick="window._ack('${esc(e.event_id)}')">Acknowledge</button></div>`;
}
window._ack = async id => { await fetch('/api/ack/' + id, {method:'POST'}); refresh(); };

let soundOn = false, lastOpen = 0, fails = 0, lastSwitch = 0;
function beep(urgent){
  if(!soundOn) return;
  try { const a = new (window.AudioContext || window.webkitAudioContext)();
    [0, 0.3].slice(0, urgent ? 2 : 1).forEach(d => { const o = a.createOscillator(), g = a.createGain();
      o.frequency.value = urgent ? 988 : 784; g.gain.value = 0.09; o.connect(g); g.connect(a.destination);
      o.start(a.currentTime + d); o.stop(a.currentTime + d + 0.22); });
  } catch(_){}
}
function setStep(id, state, text){ const el = $(id); el.className = 'step' + (state ? ' ' + state : ''); $('v' + id.slice(1)).textContent = text; }

async function refresh(){
  let s; try { s = await (await fetch('/api/state')).json(); } catch(_){ if(++fails >= 3){ document.documentElement.dataset.level = 'down'; $('status').textContent = 'Connection lost'; } return; }
  fails = 0;
  const rt = s.runtime || {};
  const watching = rt.monitoring === 'AVAILABLE';
  const open = [...(s.open_alerts || [])].sort((a,b) => info(b.type).rank - info(a.type).rank);
  if(s.open_count > lastOpen) beep(open.length && info(open[0].type).tier === 'urgent');
  lastOpen = s.open_count;
  const level = !watching ? 'down' : open.length ? (info(open[0].type).tier || 'check') : 'ok';
  document.documentElement.dataset.level = level;
  $('status').textContent = !watching ? (rt.status === 'starting' || rt.status === 'switching' ? 'Starting' : 'Not monitoring')
    : open.length ? open.length + (open.length === 1 ? ' open alert' : ' open alerts') : 'All clear';
  setHtml($('alerts'), open.map(strip).join(''));
  document.title = (open.length ? '(' + open.length + ') ' : '') + 'Live showcase';

  $('c-fps').textContent = (s.fps || 0).toFixed(1);
  $('c-people').textContent = s.people || 0;
  $('c-depth').textContent = rt.depth_active ? 'depth on' : 'no depth';
  if(rt.phase_model){ $('c-model').hidden = false; $('c-model').textContent = 'model ' + rt.phase_model.replace('.joblib',''); }
  const src = rt.source_label || rt.source || '';
  $('sub').textContent = (src ? src + ' · ' : '') + 'pose ' + (rt.backend || 'rtmo') + ' · everything runs on this device';

  const rgb = rt.show_rgb !== false;
  $('rgb').setAttribute('aria-pressed', String(rgb));
  $('rgb').textContent = rgb ? 'Camera image' : 'Stick figures only';
  $('stream').style.visibility = watching ? 'visible' : 'hidden';
  $('tag1').textContent = watching ? 'Live · ' + (s.people || 0) + ' in view' : 'No live picture';
  const dnote = !rgb ? 'Depth view is off while the camera image is off (privacy mode).'
    : !rt.depth_active ? 'This camera has no depth stream. Use the RealSense with depth on.' : '';
  $('depthnote').hidden = !dnote; $('depthnote').textContent = dnote;
  $('depth').style.visibility = dnote ? 'hidden' : 'visible';
  const dm = rt.depth_mode || 'height';
  document.querySelectorAll('[data-dmode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.dmode === dm)));
  $('tag2').textContent = dm === 'height' ? 'Height above floor, 0-2 m' : 'Distance from camera';
  $('depthcap').innerHTML = dm === 'height'
    ? '<b>Height map:</b> each pixel coloured by its height above the floor. Floor blue, mattress green, a standing head red. The system reasons in these metres, so the same rules work for any bed and any camera angle.'
    : '<b>Distance view:</b> each pixel coloured by its distance from the camera, computed on the camera itself. Black means no depth reading.';

  const tracks = s.tracks || [];
  renderBeds(s.beds || [], tracks, open, watching);

  const first = tracks.find(t => t.bed_id) || tracks[0];
  setStep('s1', watching ? 'live' : '', watching ? (s.fps || 0).toFixed(1) + ' frames per second' : 'not running');
  setStep('s2', tracks.length ? 'live' : '', tracks.length ? tracks.length + (tracks.length === 1 ? ' person' : ' people') + ', 17 joints each' : 'no one in view');
  const hh = first && (first.head_h != null || first.hip_h != null);
  setStep('s3', hh ? 'live' : '', hh ? 'head ' + (first.head_h != null ? first.head_h.toFixed(2) : '-') + ' m, hips ' + (first.hip_h != null ? first.hip_h.toFixed(2) : '-') + ' m'
    : first && first.height_m != null ? 'torso ' + first.height_m.toFixed(2) + ' m' : 'waiting for a person');
  const ph = first && first.bed_phase;
  setStep('s4', ph ? (PIDX[ph] >= 3 && PIDX[ph] <= 5 ? 'alarm' : 'live') : '', ph ? (PHASES[PIDX[ph]] || [0, ph])[1] + (first.model_phase ? ' (model: ' + ((PHASES[PIDX[first.model_phase]] || [0, first.model_phase])[1]).toLowerCase() + ')' : '') : 'no one on a bed');
  const st = first && first.state;
  setStep('s5', st ? (st === 'ON_GROUND' || st === 'FALLING' || first.possible_fall ? 'alarm' : 'live') : '', st ? (first.possible_fall ? 'possible fall' : (STATE[st] || st).toLowerCase()) : '-');
  setStep('s6', open.length ? 'alarm' : watching ? 'live' : '', open.length ? info(open[0].type).title : 'quiet: nothing needs the nurse');

  const c = s.counts || {};
  $('n-falls').textContent = Math.max(c.fall_suspected || 0, (c.fall_confirmed || 0) + (c.person_down || 0));
  $('n-possible').textContent = c.possible_fall || 0;
  $('n-exit').textContent = c.bed_exit || 0;
  $('n-up').textContent = c.bed_exit_warning || 0;
  const evs = (s.events || []).filter(e => e.alerting || (e.severity || 0) >= 2).slice(0, 25);
  setHtml($('events'), evs.length ? evs.map(e => { const a = info(e.type); return `<div class="ev ${a.tier}"><span class="mono">${esc(e.clock)}</span><span class="dot"></span>
    <span><b>${esc(a.title)}</b>${e.acknowledged ? ' &middot; acknowledged' : ''}<br/><span class="where">${esc((e.zone || e.near_bed) ? bedName(e.zone || e.near_bed) : 'In the room')}</span></span></div>`; }).join('')
    : '<div class="ev"><span></span><span></span><span class="where">Nothing yet. Try lying down, sitting up, then standing.</span></div>');
}

$('sound').addEventListener('click', function(){ soundOn = !soundOn; this.textContent = soundOn ? 'Sound on' : 'Sound off'; if(soundOn) beep(false); });
$('rgb').addEventListener('click', async function(){
  if(Date.now() - lastSwitch < 1500) return; lastSwitch = Date.now();
  const on = this.getAttribute('aria-pressed') !== 'true';
  const r = await post('/api/switch', {show_rgb:on});
  if(r && r.ok === false && r.error) alert(r.error);
  refresh();
});
document.querySelectorAll('[data-dmode]').forEach(b => b.addEventListener('click', async () => { await post('/api/depth-mode', {mode:b.dataset.dmode}); refresh(); }));
$('ackall').addEventListener('click', async () => { const s = await (await fetch('/api/state')).json(); for(const e of s.open_alerts || []) await fetch('/api/ack/' + e.event_id, {method:'POST'}); refresh(); });
const full = id => () => { const f = $(id); if(f.requestFullscreen) f.requestFullscreen(); };
$('full1').addEventListener('click', full('feed1')); $('full2').addEventListener('click', full('feed2'));
function tick(){ $('clock').textContent = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false}); }
$('depth').src = '/depth.mjpg';
refresh(); tick(); setInterval(refresh, 500); setInterval(tick, 1000);
</script>
</body>
</html>
"""
