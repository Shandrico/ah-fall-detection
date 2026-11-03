"""Count and review the live dashboard's alerts over a long run: false alarms per 24 hours.

    python scripts/alarm_audit.py --since "2026-11-04 08:00" --hours 24

1. Run `ahfd dashboard` for the whole period (overnight or longer) with normal activity in the room.
   Write down when it started and stopped; every alert goes to data/events/dashboard.jsonl and a
   stick-figure clip to data/alert_clips/.
2. Run this script. It groups alerts (one person, less than 10 s apart = one alarm, as in
   docs/BENCHMARK.md), prints counts per type, and writes data/alarm_review.csv.
3. Watch the clips (`ahfd serve-review data/alert_clips`) and fill the csv's `verdict` column:
   real / risky / false. Run the script again: it reports false alarms per 24 hours.

Only the events log and the csv are read; nothing here touches the camera.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALERTS = {"POSSIBLE_FALL", "FALL_SUSPECTED", "FALL_CONFIRMED", "PERSON_DOWN", "NEAR_MISS", "BED_EXIT", "BED_EXIT_WARNING"}
FAMILY = {"POSSIBLE_FALL": "fall", "FALL_SUSPECTED": "fall", "FALL_CONFIRMED": "fall", "PERSON_DOWN": "fall",
          "NEAR_MISS": "fall", "BED_EXIT": "bed exit", "BED_EXIT_WARNING": "early warning"}


def parse_time(text: str) -> float:
    return dt.datetime.strptime(text, "%Y-%m-%d %H:%M").timestamp()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", required=True, help='start of the monitored period, "YYYY-MM-DD HH:MM" (Jetson local time)')
    p.add_argument("--hours", type=float, required=True, help="how long the dashboard was monitoring")
    p.add_argument("--events", default=str(REPO / "data/events/dashboard.jsonl"))
    p.add_argument("--review", default=str(REPO / "data/alarm_review.csv"))
    a = p.parse_args()
    t0 = parse_time(a.since); t1 = t0 + a.hours * 3600

    rows = []
    for line in Path(a.events).read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        w = e.get("wall_time")
        if w is None or not (t0 <= w <= t1) or e.get("type") not in ALERTS:
            continue
        rows.append(e)
    rows.sort(key=lambda e: e["wall_time"])

    # One alarm = same family and person, less than 10 s after the previous alert of that alarm.
    alarms, open_ = [], {}
    for e in rows:
        key = (FAMILY[e["type"]], e.get("track_id"))
        cur = open_.get(key)
        if cur is not None and e["wall_time"] - cur["last"] < 10:
            cur["last"] = e["wall_time"]; cur["types"].append(e["type"]); continue
        cur = dict(start=e["wall_time"], last=e["wall_time"], family=key[0], track=key[1], zone=e.get("zone"), types=[e["type"]])
        open_[key] = cur; alarms.append(cur)

    review = Path(a.review)
    verdicts = {}
    if review.exists():
        with review.open() as fh:
            for r in csv.DictReader(fh):
                verdicts[r["time"]] = r.get("verdict", "").strip().lower()
    with review.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["time", "family", "types", "bed", "person", "verdict"])
        for al in alarms:
            stamp = dt.datetime.fromtimestamp(al["start"]).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([stamp, al["family"], " ".join(dict.fromkeys(al["types"])), al["zone"] or "", al["track"], verdicts.get(stamp, "")])
            al["verdict"] = verdicts.get(stamp, "")

    per24 = 24.0 / a.hours
    print(f"{len(rows)} alerts in {a.hours:g} h from {a.since} -> {len(alarms)} alarms")
    for fam in ("fall", "bed exit", "early warning"):
        fa = [x for x in alarms if x["family"] == fam]
        v = Counter(x["verdict"] or "unreviewed" for x in fa)
        print(f"  {fam:13s} {len(fa):3d} alarms ({len(fa) * per24:5.1f} per 24 h)  " + ", ".join(f"{k} {n}" for k, n in sorted(v.items())))
    false = sum(1 for x in alarms if x["verdict"] == "false")
    unrev = sum(1 for x in alarms if not x["verdict"])
    print(f"FALSE ALARMS: {false} ({false * per24:.1f} per 24 h); target 5 per 24 h. {unrev} not yet reviewed.")
    print(f"review sheet: {review}  (verdict: real / risky / false)")


if __name__ == "__main__":
    main()
