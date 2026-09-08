"""The dashboard page, as a single self-contained HTML string.

Kept in one string with no external assets so the stdlib server has nothing to
serve but this and the stream. The page loads the video once (an <img> at
/stream.mjpg -- one long-lived connection, never reopened) and polls
/api/state on a timer for the small JSON that drives the metrics, the triage
queue and the event log.

Layout borrows the useful ideas from the reference dashboard -- a metrics row,
a separate triage queue for open alerts, severity badges with evidence, an
alert sound and a fullscreen view -- and drops its multi-stream machinery,
which this single-camera build does not need.
"""

from __future__ import annotations

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Ward Fall Monitor</title>
<style>
  :root { --bg:#0d1117; --panel:#161d2b; --panel2:#0f1622; --ink:#e8eef6;
          --muted:#8695ab; --line:#243044;
          --ok:#2fb344; --warn:#f59f00; --crit:#fa5252; --accent:#3b82f6; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Segoe UI",system-ui,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { padding:12px 18px; border-bottom:1px solid var(--line);
           display:flex; justify-content:space-between; align-items:center; gap:12px; }
  h1 { font-size:17px; margin:0; letter-spacing:.3px; }
  h1 .dot { color:var(--ok); }
  .controls { display:flex; gap:8px; align-items:center; }
  select, button { background:var(--panel); color:var(--ink); border:1px solid var(--line);
                   border-radius:7px; padding:6px 10px; font-size:13px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  .metrics { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; padding:14px 14px 0; }
  @media (max-width:1100px){ .metrics{ grid-template-columns:repeat(3,1fr);} }
  .metric { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
  .metric .k { font-size:10.5px; text-transform:uppercase; letter-spacing:.7px; color:var(--muted); }
  .metric .v { font-size:26px; font-weight:700; margin-top:3px; line-height:1; }
  .metric.crit .v { color:var(--crit); }
  .metric.warn .v { color:var(--warn); }
  .wrap { display:grid; grid-template-columns:1fr 360px; gap:14px; padding:14px; }
  @media (max-width:1100px){ .wrap{ grid-template-columns:1fr; } }
  .feed { background:#000; border:1px solid var(--line); border-radius:12px; overflow:hidden; position:relative; }
  .feed img { width:100%; display:block; image-rendering:auto; }
  .feed .tag { position:absolute; left:12px; bottom:12px; background:rgba(0,0,0,.6);
               border:1px solid var(--line); border-radius:999px; padding:4px 10px; font-size:12px; }
  .feed .full { position:absolute; right:12px; top:12px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; margin-bottom:14px; }
  .panel h2 { font-size:12px; text-transform:uppercase; letter-spacing:.7px; color:var(--muted); margin:0 0 10px;
              display:flex; justify-content:space-between; }
  .chip { display:flex; justify-content:space-between; align-items:center; padding:7px 10px; border-radius:8px;
          background:var(--panel2); margin-bottom:6px; font-size:14px; }
  .badge { font-size:11px; padding:2px 9px; border-radius:999px; background:#22304a; white-space:nowrap; }
  .s-ON_GROUND,.s-FALLING { background:var(--crit); color:#fff; }
  .s-IN_BED { background:#1f6feb; color:#fff; }
  .s-SITTING { background:var(--warn); color:#000; }
  .s-UPRIGHT { background:var(--ok); color:#04210b; }
  .ev { border-left:4px solid var(--line); padding:9px 11px; margin-bottom:8px; background:var(--panel2);
        border-radius:0 8px 8px 0; }
  .ev.sev4 { border-left-color:var(--crit); } .ev.sev3 { border-left-color:var(--warn); }
  .ev.sev2 { border-left-color:var(--accent); } .ev.sev1 { border-left-color:var(--muted); }
  .ev .t { font-weight:700; font-size:14px; display:flex; justify-content:space-between; }
  .ev .t .badge { font-size:10.5px; }
  .ev .m { color:var(--muted); font-size:12px; margin-top:4px; }
  .ev .evi { color:var(--muted); font-size:11.5px; margin-top:3px; font-family:ui-monospace,monospace; }
  .ev button { margin-top:7px; padding:4px 12px; font-size:12px; }
  .ev.ack { opacity:.45; }
  .empty { color:var(--muted); font-size:13px; padding:10px; text-align:center; }
  .queue { max-height:32vh; overflow:auto; } .log { max-height:40vh; overflow:auto; }
</style>
</head>
<body>
<header>
  <h1><span class="dot">&#9679;</span> Ward Fall Monitor</h1>
  <div class="controls">
    <label style="font-size:12px;color:var(--muted)">min severity</label>
    <select id="sev">
      <option value="0">all</option>
      <option value="2">suspected+</option>
      <option value="3" selected>alerts only</option>
    </select>
    <button id="sound">&#128263; Sound: off</button>
  </div>
</header>

<section class="metrics">
  <div class="metric"><div class="k">FPS</div><div class="v" id="m-fps">-</div></div>
  <div class="metric"><div class="k">People in view</div><div class="v" id="m-people">0</div></div>
  <div class="metric crit"><div class="k">Open alerts</div><div class="v" id="m-open">0</div></div>
  <div class="metric crit"><div class="k">Confirmed falls</div><div class="v" id="m-falls">0</div></div>
  <div class="metric warn"><div class="k">Bed exits</div><div class="v" id="m-bed">0</div></div>
  <div class="metric"><div class="k">Uptime</div><div class="v" id="m-up">0s</div></div>
</section>

<div class="wrap">
  <div class="feed">
    <img id="stream" src="/stream.mjpg" alt="live view"/>
    <div class="tag" id="feedtag">live</div>
    <button class="full" id="full">Fullscreen</button>
  </div>
  <div>
    <div class="panel">
      <h2>Triage queue <span id="q-count">0</span></h2>
      <div class="queue" id="queue"><div class="empty">no open alerts</div></div>
    </div>
    <div class="panel">
      <h2>People in view</h2>
      <div id="tracks"><div class="empty">none</div></div>
    </div>
    <div class="panel">
      <h2>Event log</h2>
      <div class="log" id="events"><div class="empty">no events yet</div></div>
    </div>
  </div>
</div>

<script>
const RANK = {LOW:0, BED_EXIT:1, NEAR_MISS:1, FALL_SUSPECTED:2, PERSON_DOWN:3, FALL_CONFIRMED:4};
let soundOn = false, lastAlertCount = 0;

function esc(s){ return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function evi(e){ return e.evidence ? Object.entries(e.evidence).map(([k,v])=>k+'='+v).join('  ') : ''; }
function fmtUp(s){ s=Math.round(s); const m=Math.floor(s/60), h=Math.floor(m/60);
  return h?`${h}h${m%60}m`: m?`${m}m${s%60}s`:`${s}s`; }

function beep(){
  if(!soundOn) return;
  try { const a=new (window.AudioContext||window.webkitAudioContext)();
    const o=a.createOscillator(), g=a.createGain();
    o.type='sine'; o.frequency.value=880; g.gain.value=0.08;
    o.connect(g); g.connect(a.destination); o.start(); o.stop(a.currentTime+0.25);
  } catch(_){}
}

async function ack(id){ await fetch('/api/ack/'+id,{method:'POST'}); refresh(); }
window._ack = ack;

function card(e, withAck){
  const sev = e.severity||0;
  const btn = (withAck && sev>=3 && !e.acknowledged)
    ? `<button onclick="window._ack('${e.event_id}')">Acknowledge</button>` : '';
  return `<div class="ev sev${sev} ${e.acknowledged?'ack':''}">
    <div class="t"><span>${esc(e.type)}</span><span class="badge">track ${e.track_id}</span></div>
    <div class="m">${esc(e.clock||'')} &middot; ${esc(e.zone||'-')}</div>
    <div class="evi">${esc(evi(e))}</div>${btn}</div>`;
}

async function refresh(){
  let s; try { s = await (await fetch('/api/state')).json(); } catch(_){ return; }

  document.getElementById('m-fps').textContent = s.fps;
  document.getElementById('m-people').textContent = s.people;
  document.getElementById('m-open').textContent = s.open_count;
  document.getElementById('m-falls').textContent = s.counts.fall_confirmed + s.counts.person_down;
  document.getElementById('m-bed').textContent = s.counts.bed_exit;
  document.getElementById('m-up').textContent = fmtUp(s.uptime_s);
  document.getElementById('q-count').textContent = s.open_count;
  document.getElementById('feedtag').textContent = s.people + ' in view · ' + s.fps + ' fps';

  if (s.open_count > lastAlertCount) beep();
  lastAlertCount = s.open_count;

  const q = s.open_alerts.length
    ? s.open_alerts.map(e=>card(e,true)).join('')
    : '<div class="empty">no open alerts</div>';
  document.getElementById('queue').innerHTML = q;

  const tr = s.tracks.length ? s.tracks.map(t=>{
    const extra = (t.height_m!=null?` &middot; ${t.height_m} m`:'') + (t.zone?` &middot; ${esc(t.zone)}`:'');
    return `<div class="chip"><span>Track ${t.track_id}${extra}</span>
      <span class="badge s-${esc(t.state)}">${esc(t.state)}</span></div>`;
  }).join('') : '<div class="empty">none</div>';
  document.getElementById('tracks').innerHTML = tr;

  const floor = parseInt(document.getElementById('sev').value,10);
  const shown = s.events.filter(e=>(e.severity||0)>=floor);
  document.getElementById('events').innerHTML = shown.length
    ? shown.map(e=>card(e,true)).join('')
    : '<div class="empty">no events at this severity</div>';
}

document.getElementById('sound').addEventListener('click', function(){
  soundOn=!soundOn; this.textContent=(soundOn?'\u{1F514} Sound: on':'\u{1F508} Sound: off'); if(soundOn) beep();
});
document.getElementById('full').addEventListener('click', ()=>{
  const f=document.querySelector('.feed'); if(f.requestFullscreen) f.requestFullscreen();
});
document.getElementById('sev').addEventListener('change', refresh);

refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""
