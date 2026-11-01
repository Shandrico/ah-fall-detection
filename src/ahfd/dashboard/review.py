"""The dashboard's "Review videos" page: rendered review clips, played in the browser.

The videos hold raw RGB, so `ahfd dashboard` only passes a folder here when RGB is
authorised. Only ``.mp4`` files listed by `list_videos` are ever served: a name
from the URL is looked up in that list, never joined onto a path.
"""

from __future__ import annotations

from pathlib import Path

_UNFINISHED = (".part.mp4", ".partial.mp4")


def list_videos(root: Path | str | None) -> list[dict]:
    """The videos in ROOT's folders, as {group, name, size_mb}. Empty if ROOT is absent."""
    if not root:
        return []
    root = Path(root)
    try:
        folders = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:  # e.g. the T9 is not mounted
        return []
    videos = []
    for folder in folders:
        for path in sorted(folder.glob("*.mp4")):
            if not path.name.endswith(_UNFINISHED) and path.is_file():
                videos.append(
                    {
                        "group": folder.name,
                        "name": path.name,
                        "size_mb": round(path.stat().st_size / 1e6, 1),
                    }
                )
    return videos


def find_video(root: Path | str | None, group: str, name: str) -> Path | None:
    """The file for a listed video, or None. Anything not listed does not exist."""
    for video in list_videos(root):
        if video["group"] == group and video["name"] == name:
            return Path(root) / group / name
    return None


REVIEW_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Review Videos</title>
<style>
  :root { --bg:#0d1117; --panel:#161d2b; --panel2:#0f1622; --ink:#e8eef6; --muted:#9aa8bc;
          --line:#2d3b52; --accent:#3b82f6; --crit:#fa5252; --warn:#ff9f1c; --risk:#f5d90a; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.45 "Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--ink); }
  header { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; padding:12px 16px; border-bottom:1px solid var(--line); }
  h1 { font-size:18px; margin:0 auto 0 0; }
  a { color:#8ab8ff; }
  :focus-visible { outline:3px solid #8ab8ff; outline-offset:2px; }
  button, select { font:inherit; color:var(--ink); background:var(--panel); border:1px solid var(--line);
                   border-radius:7px; min-height:44px; padding:6px 12px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  button[aria-pressed="true"], button[aria-current="true"] { background:#1d3a66; border-color:var(--accent); }
  .wrap { display:grid; grid-template-columns:minmax(0,1fr) 320px; gap:16px; padding:16px; align-items:start; }
  @media (max-width:900px){ .wrap { grid-template-columns:minmax(0,1fr); } }
  video { display:block; width:100%; max-height:74vh; background:#000; border-radius:10px; aspect-ratio:16/9; }
  .row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
  .t { font-variant-numeric:tabular-nums; }
  .muted { color:var(--muted); font-size:13px; }
  #title { font-weight:600; margin-right:auto; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; }
  .tabs { display:flex; gap:8px; margin-bottom:10px; }
  .tabs button { flex:1; }
  #clips { max-height:62vh; overflow:auto; display:grid; gap:6px; padding:3px; }
  #clips button { width:100%; text-align:left; background:var(--panel2); }
  #clips .note { display:block; color:var(--muted); font-size:13px; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 14px; margin-top:12px; font-size:13px; }
  .swatch { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
  .empty { padding:24px 12px; text-align:center; color:var(--muted); }
  kbd { font:12px ui-monospace,monospace; border:1px solid var(--line); border-radius:4px; padding:0 4px; }
</style>
</head>
<body>
<header>
  <h1>Review videos</h1>
  <a href="/">Back to live dashboard</a>
</header>
<div class="wrap">
  <main>
    <video id="v" controls preload="metadata" aria-label="Review video"></video>
    <div class="row">
      <span id="title">Choose a clip</span>
      <span class="t muted" id="t"></span>
      <button id="cam-d435f" aria-pressed="false">D435f</button>
      <button id="cam-d435i" aria-pressed="false">D435i</button>
      <label class="muted" for="sp">Speed</label>
      <select id="sp"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>
    </div>
    <div class="row muted" id="view"></div>
    <div class="legend" aria-label="Banner colours in the video">
      <span><span class="swatch" style="background:var(--crit)"></span>FALL ALERT</span>
      <span><span class="swatch" style="background:var(--warn)"></span>POSSIBLE FALL (also an alert)</span>
      <span><span class="swatch" style="background:var(--risk)"></span>GETTING UP</span>
    </div>
    <p class="muted"><kbd>Space</kbd> play or pause &middot; <kbd>&larr;</kbd> <kbd>&rarr;</kbd> 1 s &middot; <kbd>,</kbd> <kbd>.</kbd> small step &middot; <kbd>c</kbd> other camera</p>
  </main>
  <nav class="panel" aria-label="Clips">
    <div class="tabs" id="tabs"></div>
    <div id="clips"><div class="empty">Loading&hellip;</div></div>
  </nav>
</div>
<script>
const NOTES = {f1:'Fall from a stand attempt', f3:'Trip, two falls', f4:'Slow slump', f5:'Quick recovery',
  neg_f6:'Lying down on purpose', neg_f7:'Kneel and pick-up', s1:'No exit expected', s2:'Bed exit',
  s3:'No exit expected', s4:'Bed exit'};
const TITLES = {fall:'Falls', bed_exit:'Bed exits'};
// The two cameras swapped places between the sessions.
const VIEWS = {fall:{d435f:'along the beds', d435i:'side view'}, bed_exit:{d435f:'side view', d435i:'along the beds'}};
const STEP_S = 1/15;
const $ = id => document.getElementById(id), v = $('v');
const esc = s => String(s).replace(/[&<>"']/g, c => '&#'+c.charCodeAt(0)+';');
let videos = [], group = null, clip = null, cam = 'd435f';

const parts = n => { const m = n.match(/^(.+)_(d435[fi])\.mp4$/); return m ? {clip:m[1], cam:m[2]} : {clip:n.replace(/\.mp4$/,''), cam:null}; };
const inGroup = () => videos.filter(x => x.group === group);
const clipsOf = () => [...new Set(inGroup().map(x => parts(x.name).clip))];
const current = () => inGroup().find(x => parts(x.name).clip === clip && parts(x.name).cam === cam)
                   || inGroup().find(x => parts(x.name).clip === clip);

function drawLists(){
  const groups = [...new Set(videos.map(x => x.group))];
  $('tabs').innerHTML = groups.map(g => `<button data-g="${esc(g)}" aria-pressed="${g===group}">${esc(TITLES[g]||g)}</button>`).join('');
  $('tabs').querySelectorAll('button').forEach(b => b.onclick = () => { group = b.dataset.g; clip = null; drawLists(); });
  $('clips').innerHTML = clipsOf().map(c => {
    const note = NOTES[c.replace(/_\d+$/,'')] || '';
    return `<button data-c="${esc(c)}" aria-current="${c===clip}">${esc(c)}<span class="note">${esc(note)}</span></button>`;
  }).join('');
  $('clips').querySelectorAll('button').forEach(b => b.onclick = () => choose(b.dataset.c, cam));
}

function choose(c, wanted, keepTime){
  const at = keepTime ? v.currentTime : 0, wasPlaying = keepTime ? !v.paused : true;
  clip = c; cam = wanted;
  const x = current(); if(!x) return;
  cam = parts(x.name).cam || cam;
  v.src = '/review/video/' + encodeURIComponent(x.group) + '/' + encodeURIComponent(x.name);
  v.playbackRate = parseFloat($('sp').value);
  v.onloadedmetadata = () => { if(at) v.currentTime = at; v.playbackRate = parseFloat($('sp').value); if(wasPlaying) v.play().catch(()=>{}); };
  $('title').textContent = c + (NOTES[c.replace(/_\d+$/,'')] ? ': ' + NOTES[c.replace(/_\d+$/,'')] : '');
  const view = (VIEWS[x.group]||{})[cam];
  $('view').textContent = cam.toUpperCase() + (view ? ', ' + view : '') + ' · ' + x.size_mb + ' MB';
  history.replaceState(null, '', '#' + x.group + '/' + x.name.replace(/\.mp4$/,''));
  drawLists(); drawCams();
}
function drawCams(){
  for(const c of ['d435f','d435i']){
    const b = $('cam-'+c), has = clip && inGroup().some(x => parts(x.name).clip===clip && parts(x.name).cam===c);
    b.disabled = !has; b.setAttribute('aria-pressed', String(!!clip && cam===c));
  }
}
$('cam-d435f').onclick = () => choose(clip, 'd435f', true);
$('cam-d435i').onclick = () => choose(clip, 'd435i', true);
$('sp').onchange = e => { v.playbackRate = parseFloat(e.target.value); };
v.ontimeupdate = () => { const s = v.currentTime, m = Math.floor(s/60); $('t').textContent = m + ':' + (s-60*m).toFixed(1).padStart(4,'0'); };
document.addEventListener('keydown', e => {
  if(!v.src || e.ctrlKey || e.metaKey || e.altKey || e.target.tagName === 'SELECT') return;
  if(e.key === ' ' && e.target.tagName !== 'BUTTON' && e.target !== v){ e.preventDefault(); v.paused ? v.play() : v.pause(); }
  else if(e.key === 'ArrowLeft' && e.target !== v) v.currentTime -= 1;
  else if(e.key === 'ArrowRight' && e.target !== v) v.currentTime += 1;
  else if(e.key === ','){ v.pause(); v.currentTime -= STEP_S; }
  else if(e.key === '.'){ v.pause(); v.currentTime += STEP_S; }
  else if(e.key === 'c' && clip){ choose(clip, cam==='d435f'?'d435i':'d435f', true); }
});

fetch('/api/review').then(r => r.json()).then(d => {
  videos = d.videos || [];
  if(!videos.length){
    $('clips').innerHTML = '<div class="empty">No videos found in<br/><code>'+esc(d.folder||'(no folder set)')+'</code><br/>Is the T9 mounted on the Jetson?</div>';
    return;
  }
  group = videos[0].group;
  const hash = decodeURIComponent(location.hash.slice(1)).split('/');
  const hit = videos.find(x => x.group === hash[0] && x.name === hash[1] + '.mp4');
  if(hit){ group = hit.group; const p = parts(hit.name); clip = p.clip; cam = p.cam || cam; }
  drawLists(); drawCams();
  if(hit) choose(clip, cam);
}).catch(() => { $('clips').innerHTML = '<div class="empty">Could not load the list. Is the dashboard running?</div>'; });
</script>
</body>
</html>
"""
