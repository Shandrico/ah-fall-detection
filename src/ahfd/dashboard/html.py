"""The dashboard page, as a single self-contained HTML string.

Kept in one string with no external assets so the stdlib server has nothing
to serve but this and the stream. The page loads the video stream once (an
<img> pointing at /stream.mjpg) and polls /api/state on a timer for the alert
list and per-track state -- state is small JSON, the video is the heavy part
and it is a single long-lived connection, never reopened.
"""

from __future__ import annotations

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Ward Fall Monitor</title>
<style>
  :root { --bg:#0f1420; --panel:#182031; --ink:#e7edf5; --muted:#8b97ab;
          --line:#26324a; --ok:#2f9e44; --warn:#f08c00; --crit:#e03131; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Segoe UI",system-ui,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { padding:12px 18px; border-bottom:1px solid var(--line);
           display:flex; justify-content:space-between; align-items:center; }
  h1 { font-size:17px; margin:0; letter-spacing:.3px; }
  .stat { font-size:12px; color:var(--muted); }
  .wrap { display:grid; grid-template-columns:1fr 340px; gap:14px; padding:14px; }
  @media (max-width:900px){ .wrap{ grid-template-columns:1fr; } }
  .feed { background:#000; border:1px solid var(--line); border-radius:10px;
          overflow:hidden; }
  .feed img { width:100%; display:block; }
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:10px; padding:12px; }
  .panel h2 { font-size:13px; text-transform:uppercase; letter-spacing:.6px;
              color:var(--muted); margin:0 0 10px; }
  .track { display:flex; justify-content:space-between; padding:6px 8px;
           border-radius:6px; background:#111a2b; margin-bottom:6px; font-size:14px; }
  .badge { font-size:11px; padding:2px 8px; border-radius:999px; background:#243049; }
  .ev { border-left:4px solid var(--line); padding:8px 10px; margin-bottom:8px;
        background:#111a2b; border-radius:0 8px 8px 0; }
  .ev.sev4 { border-left-color:var(--crit); }
  .ev.sev3 { border-left-color:var(--warn); }
  .ev .t { font-weight:700; }
  .ev .m { color:var(--muted); font-size:12px; margin-top:3px; }
  .ev button { margin-top:6px; font-size:12px; padding:4px 10px; cursor:pointer;
               border:1px solid var(--line); background:#1c2740; color:var(--ink);
               border-radius:6px; }
  .ev.ack { opacity:.5; }
  .empty { color:var(--muted); font-size:13px; padding:8px; }
</style>
</head>
<body>
<header>
  <h1>Ward Fall Monitor</h1>
  <div class="stat"><span id="fps">-</span> fps · <span id="open">0</span> open alerts</div>
</header>
<div class="wrap">
  <div class="feed"><img src="/stream.mjpg" alt="live view"/></div>
  <div>
    <div class="panel"><h2>People in view</h2><div id="tracks"><div class="empty">none</div></div></div>
    <div class="panel" style="margin-top:14px"><h2>Alerts</h2><div id="events"><div class="empty">no alerts</div></div></div>
  </div>
</div>
<script>
async function ack(id){ await fetch('/api/ack/'+id,{method:'POST'}); refresh(); }
function esc(s){ return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
async function refresh(){
  try {
    const s = await (await fetch('/api/state')).json();
    document.getElementById('fps').textContent = s.fps;
    document.getElementById('open').textContent = s.open_alerts;

    const tracks = s.tracks.length ? s.tracks.map(t =>
      `<div class="track"><span>Track ${t.track_id}</span><span class="badge">${esc(t.state)}</span></div>`
    ).join('') : '<div class="empty">none</div>';
    document.getElementById('tracks').innerHTML = tracks;

    const events = s.events.length ? s.events.map(e => {
      const ev = e.evidence ? Object.entries(e.evidence).map(([k,v])=>k+'='+v).join(', ') : '';
      const btn = (e.severity>=3 && !e.acknowledged) ? `<br><button onclick="ack('${e.event_id}')">Acknowledge</button>` : '';
      return `<div class="ev sev${e.severity} ${e.acknowledged?'ack':''}">
        <div class="t">${esc(e.type)} · track ${e.track_id} · ${e.t_alert}s</div>
        <div class="m">${esc(e.zone||'-')} — ${esc(ev)}</div>${btn}</div>`;
    }).join('') : '<div class="empty">no alerts</div>';
    document.getElementById('events').innerHTML = events;
  } catch (_) {}
}
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""
