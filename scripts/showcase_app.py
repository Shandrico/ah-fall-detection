"""Bed Monitor: run the dashboard as a desktop app (one icon, one full-screen window).

    python scripts/showcase_app.py                    # start, open the showcase full screen
    python scripts/showcase_app.py --page ward        # open the nurse board instead
    python scripts/showcase_app.py --window           # a normal window instead of full screen
    python scripts/showcase_app.py --install-shortcut # put a "Bed Monitor" icon on the desktop

It starts `ahfd dashboard` in the background (Jetson: configs/showcase.yaml,
laptop: configs/showcase_laptop.yaml), waits until the camera is running, then
opens the page in a browser APP window: no address bar, no tabs, its own
profile, full screen (kiosk). Closing that window (Alt+F4) stops the dashboard
and releases the camera. Everything still runs on this computer; nothing goes
over the internet. If a dashboard is already running on the port, the window
just attaches to it and leaves it running afterwards.

Log: data/logs/showcase_app.log
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "logs" / "showcase_app.log"
WINDOWS = sys.platform.startswith("win")
JETSON = sys.platform.startswith("linux") and platform.machine().lower() in ("aarch64", "arm64")
APP_NAME = "Bed Monitor"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def tell(msg: str) -> None:
    """Show a message even when started from an icon (no terminal)."""
    log(msg)
    try:
        if WINDOWS:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, msg, APP_NAME, 0x30)
        elif shutil.which("zenity"):
            subprocess.run(["zenity", "--warning", "--title", APP_NAME, "--text", msg], timeout=600)
        elif shutil.which("notify-send"):
            subprocess.run(["notify-send", APP_NAME, msg], timeout=10)
    except Exception:
        pass


def state(port: int) -> dict | None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/api/state" % port, timeout=2) as r:
            return json.load(r)
    except Exception:
        return None


# ---- the dashboard ----------------------------------------------------------

def start_dashboard(config: str, port: int, source: str | None) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "ahfd.cli", "dashboard", "--config", config, "--port", str(port)]
    if source:
        cmd += ["--source", source]
    log("starting: " + " ".join(cmd))
    out = (ROOT / "data" / "logs" / "dashboard.log").open("a", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if WINDOWS else 0  # no extra console window
    return subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env, creationflags=flags)


def wait_ready(proc: subprocess.Popen | None, port: int, timeout_s: float = 120) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            return False
        s = state(port)
        if s and (s.get("runtime") or {}).get("status") in ("running", "error", "stopped"):
            return True
        time.sleep(0.5)
    return state(port) is not None


def stop_dashboard(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    log("stopping the dashboard")
    try:
        if WINDOWS:
            proc.terminate()
        else:
            proc.send_signal(signal.SIGINT)  # lets it close the camera cleanly
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


# ---- the window ---------------------------------------------------------------

def find_browser() -> tuple[str, str] | None:
    """(kind, path): a Chromium-family browser if possible, else Firefox."""
    if WINDOWS:
        pf = [os.environ.get(k, "") for k in ("ProgramFiles(x86)", "ProgramFiles", "LocalAppData")]
        for base in pf:
            for rel in (r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"):
                p = Path(base) / rel
                if base and p.exists():
                    return "chromium", str(p)
        return None
    for name in ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable"):
        p = shutil.which(name)
        if p:
            return "chromium", p
    p = shutil.which("firefox")
    return ("firefox", p) if p else None


def profile_dir(kind: str, path: str) -> Path:
    """A browser profile of its own, so the window is a separate process we can wait on."""
    real = os.path.realpath(path)
    if not WINDOWS and ("/snap/" in real or real.startswith("/snap")):
        # Snap browsers cannot write to hidden folders in $HOME.
        d = Path.home() / "snap" / ("firefox" if kind == "firefox" else "chromium") / "common" / "ahfd-app"
    elif WINDOWS:
        d = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ahfd" / "app-profile"
    else:
        d = Path.home() / ".cache" / "ahfd" / "app-profile"
    d.mkdir(parents=True, exist_ok=True)
    return d


def open_window(url: str, kiosk: bool) -> subprocess.Popen | None:
    found = find_browser()
    if not found:
        tell("No Chromium, Chrome, Edge or Firefox found. Open " + url + " in any browser.")
        return None
    kind, path = found
    prof = profile_dir(kind, path)
    if kind == "chromium":
        cmd = [path, "--user-data-dir=" + str(prof), "--no-first-run", "--no-default-browser-check",
               "--disable-session-crashed-bubble", "--disable-infobars", "--disable-translate",
               "--autoplay-policy=no-user-gesture-required", "--overscroll-history-navigation=0"]
        cmd += (["--kiosk", url] if kiosk else ["--app=" + url, "--start-maximized"])
    else:
        cmd = [path, "--new-instance", "--profile", str(prof)] + (["--kiosk"] if kiosk else []) + [url]
    log("opening window: " + " ".join(cmd))
    return subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---- desktop icon ---------------------------------------------------------------

def make_icon() -> tuple[Path, Path | None]:
    """A simple app icon: a bed with a person getting up, on the ward-board teal."""
    from PIL import Image, ImageDraw

    png, ico = ROOT / "assets" / "bed_monitor.png", ROOT / "assets" / "bed_monitor.ico"
    png.parent.mkdir(exist_ok=True)
    s = 256
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([8, 8, s - 8, s - 8], radius=52, fill=(14, 59, 61, 255))
    w = (246, 242, 232, 255)
    d.rectangle([44, 70, 58, 196], fill=w)                         # headboard
    d.rounded_rectangle([58, 140, 200, 162], radius=6, fill=w)     # mattress
    d.rectangle([58, 162, 210, 172], fill=w)                       # frame
    d.rectangle([68, 172, 78, 200], fill=w); d.rectangle([190, 172, 200, 200], fill=w)
    d.rounded_rectangle([66, 122, 104, 140], radius=8, fill=(169, 196, 191, 255))  # pillow
    d.ellipse([168, 64, 204, 100], fill=(242, 163, 15, 255))      # head of someone sitting up at the edge
    d.line([186, 100, 182, 140], fill=(242, 163, 15, 255), width=16)
    d.line([182, 140, 214, 146], fill=(242, 163, 15, 255), width=14)
    d.line([214, 146, 216, 196], fill=(242, 163, 15, 255), width=14)
    im.save(png)
    try:
        im.save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    except Exception:
        ico = None
    return png, ico


def install_shortcut() -> None:
    png, ico = make_icon()
    script = ROOT / "scripts" / "showcase_app.py"
    if WINDOWS:
        pyw = Path(sys.executable).with_name("pythonw.exe")
        target = pyw if pyw.exists() else Path(sys.executable)
        desktop = Path(os.environ.get("USERPROFILE", Path.home())) / "Desktop"
        entries = [(APP_NAME, ""), (APP_NAME + " - nurse board", " --page ward")]
        for name, extra in entries:
            lnk = desktop / (name + ".lnk")
            ps = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
                  "$s.TargetPath='{t}';$s.Arguments='\"{sc}\"{x}';$s.WorkingDirectory='{wd}';"
                  "$s.IconLocation='{ic}';$s.Description='{n}';$s.Save()").format(
                lnk=lnk, t=target, sc=script, x=extra, wd=ROOT, ic=ico or png, n=name)
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
            log("shortcut: " + str(lnk))
        return
    py = sys.executable
    body = "\n".join([
        "[Desktop Entry]", "Type=Application", "Version=1.0", "Name=" + APP_NAME,
        "Comment=Bed-exit and fall monitor (live showcase)",
        'Exec="%s" "%s"' % (py, script), "Path=" + str(ROOT), "Icon=" + str(png),
        "Terminal=false", "Categories=Utility;", "StartupNotify=true", "Actions=ward;window;", "",
        "[Desktop Action ward]", "Name=Nurse board", 'Exec="%s" "%s" --page ward' % (py, script), "",
        "[Desktop Action window]", "Name=Showcase in a window", 'Exec="%s" "%s" --window' % (py, script), "",
    ])
    targets = [Path.home() / ".local" / "share" / "applications" / "bed-monitor.desktop"]
    if (Path.home() / "Desktop").is_dir():
        targets.append(Path.home() / "Desktop" / "bed-monitor.desktop")
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(body, encoding="utf-8")
        t.chmod(0o755)
        if shutil.which("gio"):  # GNOME: allow launching without the "untrusted" prompt
            subprocess.run(["gio", "set", str(t), "metadata::trusted", "true"], check=False)
        log("shortcut: " + str(t))


# ---- main -------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", default="showcase", choices=["showcase", "ward", "engineer"])
    ap.add_argument("--window", action="store_true", help="a normal window instead of full screen")
    ap.add_argument("--config", default=None, help="dashboard config (default: showcase for this machine)")
    ap.add_argument("--source", default=None, help="override the camera/source URI")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--install-shortcut", action="store_true", help="create the desktop icon(s) and exit")
    args = ap.parse_args()
    os.chdir(ROOT)

    if args.install_shortcut:
        install_shortcut()
        return 0

    config = args.config or ("configs/showcase.yaml" if JETSON else "configs/showcase_laptop.yaml")
    url = "http://127.0.0.1:%d/%s" % (args.port, "" if args.page == "engineer" else args.page)
    log("---- %s: %s, %s" % (APP_NAME, config, url))

    proc = None
    if state(args.port) is not None:
        log("a dashboard is already running on port %d; attaching to it" % args.port)
    else:
        proc = start_dashboard(config, args.port, args.source)
        if not wait_ready(proc, args.port):
            tail = ""
            try:
                tail = "\n".join((ROOT / "data/logs/dashboard.log").read_text(encoding="utf-8", errors="replace").splitlines()[-8:])
            except Exception:
                pass
            if proc.poll() is not None:
                tell("The monitor could not start. Is the camera plugged in (USB 3, blue port)?\n\n" + tail)
                return 1
            log("camera still starting; opening the window anyway")

    win = open_window(url, kiosk=not args.window)
    try:
        if win is None:
            if proc is not None:
                proc.wait()
            return 0
        t0 = time.time()
        while win.poll() is None:
            if proc is not None and proc.poll() is not None:
                tell("The monitor stopped unexpectedly. See data/logs/dashboard.log, then start it again.")
                win.terminate()
                return 1
            time.sleep(0.5)
        if time.time() - t0 < 3 and proc is not None:
            # The browser handed the window to another instance; keep serving until stopped.
            log("window handed off to a running browser; press Ctrl+C here to stop the monitor")
            proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if proc is not None:
            stop_dashboard(proc)
    log("closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
