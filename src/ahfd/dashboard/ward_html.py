"""The ward page (/ward): what a nurse needs, like a board at the nurses' station.

The engineer dashboard at / keeps every control and number; this page has none of
them. One self-contained string with no outside assets, so a ward needs no internet.

In order of attention: is monitoring working, is anyone in trouble, then the camera,
which shows the room best, beside one status row per bed. Every status has words
as well as colour and a small drawing. The top band takes the
colour of the worst open alert.

"All clear" is only ever shown while monitoring is really running: a stopped camera
or a lost connection must never read as a quiet ward.
"""

from __future__ import annotations

WARD_HTML = r"""<!doctype html>
<html lang="en" data-theme="day" data-level="ok">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Ward Board</title>
<style>
  :root { color-scheme:light;
    --paper:#f2eee5; --board:#fbf9f3; --ink:#18282b; --soft:#5a6866; --rule:#cbc5b6;
    --sign:#0e3b3d; --on-sign:#f6f2e8; --sign-soft:#a9c4bf;
    --red:#c4261c; --on-red:#ffffff; --red-wash:#f8ddd8;
    --amber:#f2a30f; --on-amber:#1d1503; --amber-wash:#fbeccb; --amber-ink:#7d4e00;
    --go:#17694b; --go-wash:#dcebdf;
    --mattress:#ded8c8; --frame:#55646a; --blanket:#7ba7a0; --pillow:#fbf9f3; --figure:#18282b; }
  [data-theme="night"] { color-scheme:dark;
    --paper:#0c1416; --board:#141f22; --ink:#e9efec; --soft:#9fb1ae; --rule:#2c3b3e;
    --sign:#0a2527; --on-sign:#e9efec; --sign-soft:#8fb0aa;
    --red:#c4261c; --on-red:#ffffff; --red-wash:#3c1714;
    --amber:#f2a30f; --on-amber:#1d1503; --amber-wash:#35290d; --amber-ink:#ffc660;
    --go:#6fd3a6; --go-wash:#12291f;
    --mattress:#33464a; --frame:#7d9094; --blanket:#3f6a65; --pillow:#57696c; --figure:#e9efec; }
  * { box-sizing:border-box; }
  [hidden] { display:none !important; }
  html { background:var(--paper); }
  body { margin:0; color:var(--ink); background:var(--paper);
         font:17px/1.35 Bahnschrift,"DIN Alternate","Avenir Next","Roboto Condensed","Segoe UI",system-ui,sans-serif; }
  h1,h2,h3,p { margin:0; font-weight:inherit; font-size:inherit; }
  :focus-visible { outline:3px solid currentColor; outline-offset:3px; }
  button { font:inherit; color:inherit; background:none; border:2px solid currentColor; border-radius:3px;
           min-height:44px; padding:6px 14px; cursor:pointer; }
  a { color:inherit; }
  .time { font-family:ui-monospace,"Cascadia Mono",Consolas,monospace; font-variant-numeric:tabular-nums; }

  /* The sign over the ward: its colour is the worst thing happening. */
  .band { display:flex; flex-wrap:wrap; align-items:center; gap:8px 26px; padding:12px 24px;
          background:var(--sign); color:var(--on-sign); }
  .band .clock { font-size:44px; line-height:1; letter-spacing:-.02em; }
  .band h1 { font-size:15px; letter-spacing:.14em; text-transform:uppercase; font-weight:600; }
  .band .state { font-size:22px; font-weight:700; }
  .band .actions { margin-left:auto; display:flex; flex-wrap:wrap; gap:10px; align-items:center; font-size:15px; }
  .band .actions a { padding:0 6px; }
  #sound.off { background:var(--amber); color:var(--on-amber); border-color:var(--amber); font-weight:700; }
  [data-level="urgent"] .band { background:var(--red); color:var(--on-red); }
  [data-level="check"] .band { background:var(--amber); color:var(--on-amber); }
  [data-level="check"] #sound.off { background:var(--on-amber); color:var(--amber); border-color:var(--on-amber); }
  [data-level="down"] .band { color:#fff; background:#111;
    background-image:repeating-linear-gradient(135deg,#111 0 22px,#3a2a05 22px 44px); }

  /* Alerts: solid, wide, unmissable. */
  .strip { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:10px 28px; align-items:center;
           padding:8px 24px; border-bottom:2px solid var(--paper); }
  .strip.urgent { background:var(--red); color:var(--on-red); }
  .strip.check { background:var(--amber); color:var(--on-amber); }
  .strip .line { display:flex; flex-wrap:wrap; align-items:baseline; gap:2px 16px; }
  .strip .tier { font-size:14px; letter-spacing:.14em; text-transform:uppercase; font-weight:700; }
  .strip h2 { font-size:32px; line-height:1.1; font-weight:700; }
  .strip .where { font-size:20px; }
  .strip .open { margin-left:12px; opacity:.9; font-size:16px; }
  .strip .up { font-size:17px; font-weight:700; margin-top:2px; }
  .strip button { background:#fff; color:#18282b; border-color:#fff; font-weight:700; font-size:18px; min-height:48px; padding:6px 26px; }
  .strip.check button { background:#18282b; color:#fff; border-color:#18282b; }
  .quiet { display:flex; flex-wrap:wrap; align-items:baseline; gap:4px 16px; padding:12px 24px;
           border-bottom:2px solid var(--rule); }
  .quiet strong { font-size:24px; }
  .quiet.ok { background:var(--go-wash); color:var(--go); }
  .quiet.ok span { color:var(--ink); }
  .quiet.wait { background:var(--amber-wash); color:var(--amber-ink); }
  .quiet.wait span { color:var(--ink); }
  .quiet.down { background:var(--red); color:#fff; padding:18px 24px; }
  .quiet.down strong { font-size:34px; }
  .quiet.down span { font-size:19px; }

  .layout { display:grid; grid-template-columns:minmax(0,2.1fr) minmax(340px,1fr); gap:24px; padding:18px 24px 30px; align-items:start; }
  @media (max-width:1050px){ .layout { grid-template-columns:minmax(0,1fr); } .side { order:-1; } }
  .stack { display:grid; gap:22px; min-width:0; }
  .label { display:flex; justify-content:space-between; gap:12px; font-size:14px; font-weight:700; letter-spacing:.14em;
           text-transform:uppercase; border-top:3px solid var(--ink); padding-top:7px; margin-bottom:12px; }
  .label small { font-size:14px; font-weight:400; letter-spacing:0; text-transform:none; color:var(--soft); }

  /* One row per bed: the status first, the setting quietly under it. */
  .beds { display:grid; gap:10px; }
  .bed { display:grid; grid-template-columns:128px minmax(0,1fr); gap:6px 14px; align-items:center; background:var(--board);
         border:1px solid var(--rule); border-left:10px solid var(--rule); border-radius:3px; padding:10px 12px; min-width:0; }
  .bed .art svg { display:block; width:100%; height:auto; }
  .bed .bedname { font-size:14px; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); font-weight:600; }
  .bed .now { font-size:27px; line-height:1.1; font-weight:700; }
  .bed.urgent { border-color:var(--red); background:var(--red-wash); }
  .bed.urgent .now { color:var(--red); }
  .bed.check { border-color:var(--amber); background:var(--amber-wash); }
  .bed.check .now { color:var(--amber-ink); }
  [data-theme="night"] .bed.urgent .now { color:#ff8b7f; }
  .bed.off .art { opacity:.35; }
  .bed.off .now { color:var(--soft); }
  .chart { grid-column:1 / -1; display:flex; align-items:center; justify-content:space-between; gap:10px; }
  .chart .cap { font-size:14px; color:var(--soft); }
  .levels { display:flex; }
  .levels button { border-width:1px; border-color:var(--soft); border-radius:0; margin-left:-1px; min-width:64px; padding:4px 8px; font-size:15px; }
  .levels button[aria-pressed="true"] { background:var(--ink); color:var(--board); border-color:var(--ink); font-weight:700; }
  .key { margin-top:8px; font-size:14px; color:var(--soft); }
  .nobeds { padding:20px 8px; color:var(--soft); border:2px dashed var(--rule); text-align:center; }

  /* The shift log. */
  .row { display:grid; grid-template-columns:84px 12px minmax(0,1fr); gap:8px; align-items:baseline; padding:9px 0; border-bottom:1px solid var(--rule); }
  .row .mark { width:12px; height:12px; align-self:center; background:var(--rule); }
  .row.urgent .mark { background:var(--red); }
  .row.check .mark { background:var(--amber); transform:rotate(45deg); }
  .row .what { font-weight:600; }
  .row .place { color:var(--soft); font-size:15px; }
  .row.done .what { font-weight:400; }
  .none { color:var(--soft); padding:6px 0; }

  .feed { position:relative; background:#050708; overflow:hidden; }
  .feed img { display:block; width:100%; aspect-ratio:16/9; max-height:74vh; object-fit:contain; }
  #log { max-height:30vh; overflow:auto; }
  .feed:fullscreen { display:flex; align-items:center; }
  .feed .tag { position:absolute; left:0; bottom:0; padding:4px 10px; background:#050708; color:#fff; font-size:14px; }
  .feed button { position:absolute; right:8px; top:8px; background:#050708; color:#fff; border-width:1px; font-size:14px; }
  .tally { display:grid; grid-template-columns:repeat(4,1fr); }
  .tally div { padding:0 10px; border-left:1px solid var(--rule); }
  .tally div:first-child { border-left:0; padding-left:0; }
  .tally b { display:block; font-size:32px; line-height:1; font-weight:700; }
  .tally span { font-size:14px; color:var(--soft); }
  .tally .hot b { color:var(--red); }
  [data-theme="night"] .tally .hot b { color:#ff8b7f; }
  .person { display:flex; justify-content:space-between; gap:10px; padding:8px 0; border-bottom:1px solid var(--rule); }
  .person b { font-weight:700; }
  @media (max-width:640px){
    .band, .quiet, .strip { padding-left:16px; padding-right:16px; }
    .layout { padding:18px 16px 28px; }
    .band .actions { margin-left:0; }
    .strip { grid-template-columns:minmax(0,1fr); }
    .strip h2 { font-size:32px; }
    .strip button { width:100%; }
  }
  @media (prefers-reduced-motion:reduce){ * { transition:none !important; } }
</style>
</head>
<body>
<header class="band">
  <div class="clock time" id="clock"></div>
  <div><h1>Ward board</h1><div class="state" id="state" role="status">Starting</div></div>
  <div class="actions">
    <button id="sound" class="off">Turn alert sound on</button>
    <button id="theme" aria-pressed="false">Night</button>
    <a href="/">Engineer view</a>
  </div>
</header>

<section id="alerts" aria-label="Alerts" aria-live="assertive"></section>

<div class="layout">
  <section class="stack" aria-label="Camera">
    <div>
      <div class="label">Camera <small id="privacy"></small></div>
      <div class="feed" id="feed">
        <img id="stream" src="/stream.mjpg" alt="" aria-label="Live camera view"/>
        <div class="tag" id="feedtag">Connecting</div>
        <button id="full">Fullscreen</button>
      </div>
    </div>
  </section>
  <div class="stack side">
    <section aria-label="Beds">
      <div class="label">Beds</div>
      <div class="beds" id="beds"><div class="nobeds">No beds set up for this camera</div></div>
      <p class="key">Alert level. Low: falls only. Medium: also when the patient has left the bed. High: also when they start getting up.</p>
    </section>
    <section aria-label="Elsewhere in the room" id="elsewhere-box" hidden>
      <div class="label">Elsewhere in the room</div>
      <div id="elsewhere"></div>
    </section>
    <section aria-label="Shift log">
      <div class="label">Shift log <small>newest first</small></div>
      <div id="log"><div class="none">Nothing yet</div></div>
    </section>
    <section aria-label="Totals">
      <div class="label">Last 24 hours</div>
      <div class="tally">
        <div id="falls-box"><b id="n-falls">0</b><span>Falls</span></div>
        <div><b id="n-possible">0</b><span>Possible falls</span></div>
        <div><b id="n-exit">0</b><span>Left bed</span></div>
        <div><b id="n-up">0</b><span>Getting up</span></div>
      </div>
    </section>
  </div>
</div>

<script>
const ALERTS = {
  FALL_CONFIRMED:   {rank:6, tier:'urgent', label:'Urgent', title:'Fall: still on the floor'},
  PERSON_DOWN:      {rank:5, tier:'urgent', label:'Urgent', title:'Person on the floor'},
  FALL_SUSPECTED:   {rank:4, tier:'urgent', label:'Urgent', title:'Fall detected'},
  POSSIBLE_FALL:    {rank:3, tier:'check',  label:'Check now', title:'Possible fall'},
  NEAR_MISS:        {rank:2, tier:'check',  label:'Check now', title:'Fell and got back up'},
  BED_EXIT:         {rank:2, tier:'check',  label:'Check now', title:'Patient left the bed'},
  BED_EXIT_WARNING: {rank:1, tier:'check',  label:'Check soon', title:'Patient may be getting up'},
};
const WORDS = {UPRIGHT:'Standing', SITTING:'Sitting', IN_BED:'In bed', FALLING:'Falling', ON_GROUND:'On the floor',
  RECOVERED:'Got back up', LOW_CONFIDENCE:'Cannot see clearly', UNKNOWN:'Cannot tell', TRACKED:'In view'};
const LEVELS = {low:'Alerts for falls only.', medium:'Also alerts when the patient has left the bed.',
  high:'Also warns when the patient starts getting up.'};

const POSES = {
  none: '<rect x="150" y="72" width="62" height="13" rx="3" fill="var(--blanket)"/>',
  lying: '<circle cx="62" cy="61" r="17" fill="var(--figure)"/><path d="M78 70 L112 73" class="fig"/><path d="M92 84 C96 58 196 58 214 84 Z" fill="var(--blanket)"/>',
  situp: '<circle cx="70" cy="26" r="17" fill="var(--figure)"/><path d="M74 42 L92 78" class="fig"/><path d="M92 78 L200 76" class="fig"/><path d="M104 84 C108 62 196 62 214 84 Z" fill="var(--blanket)"/>',
  edge: '<circle cx="222" cy="24" r="17" fill="var(--figure)"/><path d="M222 40 L218 80 L250 82 L250 142 L266 144" class="fig"/><path d="M221 50 L240 74" class="fig thin"/>',
  stand: '<circle cx="270" cy="14" r="17" fill="var(--figure)"/><path d="M270 30 L270 88 M270 88 L258 148 M270 88 L284 148 M270 40 L254 76 M270 40 L288 74" class="fig"/>',
  falling: '<circle cx="318" cy="66" r="17" fill="var(--figure)"/><path d="M304 76 L262 112 M262 112 L232 146 M262 112 L286 148 M296 82 L306 116 M296 82 L336 96" class="fig"/>',
  floor: '<circle cx="238" cy="136" r="17" fill="var(--figure)"/><path d="M254 142 L314 145 L338 128 L352 146 M268 142 L292 124" class="fig"/>',
  unknown: '<text x="285" y="112" text-anchor="middle" font-size="60" font-weight="700" fill="var(--soft)">?</text>',
};
function bedDrawing(pose){
  return `<svg viewBox="0 0 360 164" role="img" aria-hidden="true">
    <style>.fig{fill:none;stroke:var(--figure);stroke-width:14;stroke-linecap:round;stroke-linejoin:round}.thin{stroke-width:11}</style>
    <path d="M6 156 H354" stroke="var(--frame)" stroke-width="3"/>
    <rect x="20" y="48" width="10" height="100" fill="var(--frame)"/>
    <rect x="30" y="106" width="190" height="9" fill="var(--frame)"/>
    <path d="M52 115 V144 M198 115 V144" stroke="var(--frame)" stroke-width="7"/>
    <circle cx="52" cy="148" r="7" fill="var(--frame)"/><circle cx="198" cy="148" r="7" fill="var(--frame)"/>
    <rect x="30" y="84" width="190" height="22" rx="4" fill="var(--mattress)"/>
    <rect x="34" y="72" width="50" height="14" rx="6" fill="var(--pillow)" stroke="var(--frame)" stroke-width="1"/>
    ${POSES[pose] || ''}
  </svg>`;
}

let soundOn = false, lastOpen = 0, failedPolls = 0, bedKey = '';
const firstSeen = {};
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => '&#' + c.charCodeAt(0) + ';');
// Rewriting a list every second would drop keyboard focus from its buttons.
function setHtml(el, html){ if(el._html !== html){ el._html = html; el.innerHTML = html; } }
function store(k, v){ try { localStorage.setItem(k, v); } catch(_){} }
function stored(k){ try { return localStorage.getItem(k); } catch(_){ return null; } }
function bedName(id){
  const name = String(id).replace(/_/g, ' ');
  const cap = name.charAt(0).toUpperCase() + name.slice(1);
  return /bed/i.test(cap) ? cap : cap + ' bed';
}
const place = zone => zone ? bedName(zone) : 'See camera for location';
const info = type => ALERTS[type] || {rank:0, tier:'check', label:'Check', title:String(type).replace(/_/g,' ')};
const people = n => n === 0 ? 'No one in view' : n === 1 ? '1 person in view' : n + ' people in view';
function minutes(s){ const m = Math.floor(s/60); return m < 1 ? 'under a minute' : m === 1 ? '1 minute' : m < 90 ? m + ' minutes' : Math.round(m/60) + ' hours'; }

function setTheme(t){
  document.documentElement.dataset.theme = t;
  $('theme').textContent = t === 'night' ? 'Day' : 'Night';
  store('ahfd-ward-theme', t);
}
function beep(){
  if(!soundOn) return;
  try { const a = new (window.AudioContext||window.webkitAudioContext)();
    const o = a.createOscillator(), g = a.createGain();
    o.type = 'sine'; o.frequency.value = 880; g.gain.value = 0.08;
    o.connect(g); g.connect(a.destination); o.start(); o.stop(a.currentTime + 0.25);
  } catch(_){}
}
async function post(path, body){
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  try { return await r.json(); } catch(_){ return {ok:false}; }
}
window._ack = async id => { await fetch('/api/ack/' + id, {method:'POST'}); refresh(); };

// What a person is doing: the words, the drawing, and how worried to be.
function reading(t){
  const phase = t.bed_phase, st = t.state;
  if(st === 'ON_GROUND') return {text:'On the floor', pose:'floor', tier:'urgent'};
  if(st === 'FALLING') return {text:'Falling', pose:'falling', tier:'urgent'};
  if(t.possible_fall) return {text:'Possible fall', pose:'falling', tier:'check'};
  if(phase === 'ATTEMPTING_STAND') return {text:'Trying to stand', pose:'stand', tier:'check'};
  if(phase === 'EDGE_SITTING' || phase === 'SHIFTING_TO_EDGE') return {text:'At the bed edge', pose:'edge', tier:'check'};
  if(phase === 'TORSO_RISING') return {text:'Sitting up', pose:'situp', tier:'check'};
  if(phase === 'OUT_OF_BED') return {text:'Out of bed', pose:'stand', tier:'check'};
  if(st === 'IN_BED' || phase === 'RECLINED') return {text:'In bed', pose:'lying', tier:''};
  if(st === 'SITTING' || phase === 'UPRIGHT_IN_BED') return {text:'Sitting up', pose:'situp', tier:''};
  if(st === 'UPRIGHT' || st === 'RECOVERED') return {text:WORDS[st], pose:'stand', tier:''};
  return {text:WORDS[st] || 'In view', pose:'unknown', tier:''};
}

function renderBeds(beds, tracks, open, watching){
  const box = $('beds');
  const key = JSON.stringify(beds);
  if(key !== bedKey){
    bedKey = key;
    box.innerHTML = beds.length ? beds.map(b => `<article class="bed" data-bed="${esc(b.bed_id)}">
        <div class="art"></div>
        <div><div class="bedname">${esc(bedName(b.bed_id))}</div><div class="now"></div></div>
        <div class="chart">
          <span class="cap">Alert level</span>
          <div class="levels" role="group" aria-label="${esc('Alert level for ' + bedName(b.bed_id))}" title="${esc(LEVELS[b.mode] || '')}">
            ${[['low','Low'],['medium','Medium'],['high','High']].map(([m,w]) =>
              `<button data-level="${m}" aria-pressed="${b.mode === m}">${w}</button>`).join('')}
          </div>
        </div></article>`).join('') : '<div class="nobeds">No beds set up for this camera</div>';
    box.querySelectorAll('.bed').forEach(card => card.querySelectorAll('button').forEach(btn =>
      btn.addEventListener('click', async () => { await post('/api/bed-mode', {bed_id:card.dataset.bed, mode:btn.dataset.level}); bedKey = ''; refresh(); })));
  }
  box.querySelectorAll('.bed').forEach(card => {
    const id = card.dataset.bed;
    const here = tracks.filter(t => t.bed_id === id).map(reading);
    const alerts = open.filter(e => (e.zone || e.near_bed) === id).map(e => info(e.type)).sort((a,b) => b.rank - a.rank);
    const top = here.find(x => x.tier === 'urgent') || here.find(x => x.tier === 'check') || here[0];
    const urgent = (top && top.tier === 'urgent') || alerts.some(a => a.tier === 'urgent');
    const check = !urgent && ((top && top.tier === 'check') || alerts.length > 0);
    card.className = 'bed' + (urgent ? ' urgent' : check ? ' check' : '') + (watching ? '' : ' off');
    card.querySelector('.now').textContent = !watching ? 'Not being monitored'
      : top ? top.text : alerts.length ? alerts[0].title : 'No one detected';
    const pose = !watching ? 'none' : top ? top.pose : 'none';
    const art = card.querySelector('.art');
    if(art.dataset.pose !== pose){ art.dataset.pose = pose; art.innerHTML = bedDrawing(pose); }
  });
}

function strip(e){
  const a = info(e.type);
  firstSeen[e.event_id] = firstSeen[e.event_id] || Date.now();
  return `<article class="strip ${a.tier}">
    <div><div class="line"><span class="tier">${esc(a.label)}</span><h2>${esc(a.title)}</h2>
      <span class="where">${esc(place(e.zone || e.near_bed))} &middot; at <span class="time">${esc(e.clock || '')}</span><span class="open time" data-since="${esc(e.event_id)}"></span></span></div>
      ${e.recovered_clock ? `<div class="up">Up again at ${esc(e.recovered_clock)}. Check on them, then acknowledge.</div>` : ''}
    </div>
    <button onclick="window._ack('${esc(e.event_id)}')">Acknowledge</button>
  </article>`;
}
function logRow(e){
  const a = info(e.type);
  const note = e.acknowledged ? 'Acknowledged' : e.alerting ? 'Open' : '';
  return `<div class="row ${a.tier} ${e.acknowledged ? 'done' : ''}">
    <span class="time">${esc(e.clock || '')}</span><span class="mark"></span>
    <span><span class="what">${esc(a.title)}</span><br/><span class="place">${esc(place(e.zone || e.near_bed))}${e.recovered_clock ? ', up again at ' + esc(e.recovered_clock) : ''}${note ? ' \u00b7 ' + note : ''}</span></span></div>`;
}

function monitorOf(rt, s){
  if(rt.status === 'switching' || rt.status === 'starting' || rt.monitoring_reason === 'WAITING_FOR_FRAME')
    return {cls:'wait', state:'Starting', head:'Starting up', body:'Alerts begin once the camera is running.'};
  if(rt.monitoring !== 'AVAILABLE')
    return {cls:'down', state:'NOT MONITORING', head:'Not monitoring', body:'The camera has stopped, so no alerts can be raised. Check on patients and call for technical help.'};
  if(rt.detect === false)
    return {cls:'down', state:'FALLS NOT MONITORED', head:'Falls are not being monitored', body:'The camera is on but fall detection is off for it. Call for technical help.'};
  const quiet = s.seconds_since_alert == null ? 'No alerts since this screen opened.' : 'No alerts for ' + minutes(s.seconds_since_alert) + '.';
  return {cls:'ok', state:'Monitoring', head:'All clear', body:quiet + ' ' + people(s.people) + '.'};
}
const quietBar = m => `<div class="quiet ${m.cls}"><strong>${esc(m.head)}</strong><span>${esc(m.body)}</span></div>`;

function lost(){
  const m = {cls:'down', state:'CONNECTION LOST', head:'Connection lost', body:'This screen cannot reach the monitor, so it cannot show alerts. Check on patients and call for technical help.'};
  document.documentElement.dataset.level = 'down';
  $('state').textContent = m.state;
  setHtml($('alerts'), quietBar(m));
  document.title = 'NOT CONNECTED - Ward Board';
}

async function refresh(){
  let s;
  try { s = await (await fetch('/api/state')).json(); }
  catch(_){ if(++failedPolls >= 3) lost(); return; }   // one missed poll is a blip
  failedPolls = 0;
  const rt = s.runtime || {};
  const m = monitorOf(rt, s);
  const watching = m.cls === 'ok';
  const open = [...s.open_alerts].sort((a,b) => info(b.type).rank - info(a.type).rank);

  if(s.open_count > lastOpen) beep();
  lastOpen = s.open_count;
  document.documentElement.dataset.level = m.cls === 'down' ? 'down' : open.length ? info(open[0].type).tier : m.cls;
  $('state').textContent = open.length && m.cls !== 'down'
    ? open.length + (open.length === 1 ? ' open alert' : ' open alerts') : m.state;
  // Open alerts always show. A monitoring problem is stated above them; "All clear" stands alone.
  setHtml($('alerts'), open.length ? (watching ? '' : quietBar(m)) + open.map(strip).join('') : quietBar(m));
  document.title = (open.length ? '(' + open.length + ') ALERT - ' : m.cls === 'down' ? 'NOT MONITORING - ' : '') + 'Ward Board';

  renderBeds(s.beds || [], s.tracks, open, watching);
  setHtml($('log'), s.events.length ? s.events.filter(e => e.alerting || (e.severity||0) >= 3).map(logRow).join('') || '<div class="none">Nothing yet</div>' : '<div class="none">Nothing yet</div>');
  const away = s.tracks.filter(t => !t.bed_id);
  $('elsewhere-box').hidden = !(watching && away.length);
  setHtml($('elsewhere'), away.map(t => `<div class="person"><span>Person</span><b>${esc(reading(t).text)}</b></div>`).join(''));

  const day = s.counts_24h || {}, n = t => day[t] || 0;
  const falls = Math.max(n('FALL_SUSPECTED'), n('FALL_CONFIRMED') + n('PERSON_DOWN'));  // one fall raises several alerts
  $('n-falls').textContent = falls;
  $('falls-box').className = falls ? 'hot' : '';
  $('n-possible').textContent = n('POSSIBLE_FALL');
  $('n-exit').textContent = n('BED_EXIT');
  $('n-up').textContent = n('BED_EXIT_WARNING');
  // A frozen last frame looks exactly like a live one, so never leave it up.
  $('stream').style.visibility = rt.monitoring === 'AVAILABLE' ? 'visible' : 'hidden';
  $('feedtag').textContent = rt.monitoring === 'AVAILABLE' ? 'Live · ' + people(s.people) : 'No live picture';
  $('privacy').textContent = rt.show_rgb === false ? 'Stick figures only, for privacy' : '';
  tick();
}
function tick(){
  const now = new Date();
  $('clock').textContent = now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', hour12:false});
  document.querySelectorAll('[data-since]').forEach(el => {
    const secs = Math.floor((Date.now() - (firstSeen[el.dataset.since] || Date.now())) / 1000);
    el.textContent = 'open ' + Math.floor(secs/60) + ':' + String(secs%60).padStart(2,'0');
  });
}

$('sound').addEventListener('click', function(){
  soundOn = !soundOn;
  this.textContent = soundOn ? 'Alert sound on' : 'Turn alert sound on';
  this.classList.toggle('off', !soundOn);
  if(soundOn) beep();
});
$('theme').addEventListener('click', () => setTheme(document.documentElement.dataset.theme === 'night' ? 'day' : 'night'));
$('full').addEventListener('click', () => { const f = $('feed'); if(f.requestFullscreen) f.requestFullscreen(); });
setTheme(stored('ahfd-ward-theme') === 'night' ? 'night' : 'day');
refresh(); tick();
setInterval(tick, 1000);
setInterval(refresh, 1000);
</script>
</body>
</html>
"""
