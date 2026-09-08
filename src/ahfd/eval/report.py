"""Render an evaluation Report as Markdown.

The layout puts false-alarms-per-hour first and in bold, then recall, then the
per-scenario breakdown. That order is a claim about what matters: the breakdown
table -- which scenario produced each false positive -- is what actually drives
the next round of tuning, far more than a single F1 number.
"""

from __future__ import annotations

from ahfd.eval.metrics import Report


def _fmt(value: float, spec: str = ".3f") -> str:
    return "n/a" if value != value else format(value, spec)  # nan-safe


def render_markdown(report: Report, title: str = "Fall detection evaluation") -> str:
    lines: list[str] = []
    add = lines.append

    add("# " + title)
    add("")
    add(
        "Footage: "
        + _fmt(report.total_hours, ".2f")
        + " camera-hours across "
        + str(len(report.results))
        + " clips."
    )
    add("")

    add("## Headline")
    add("")
    add(
        "**False alarms per camera-hour: "
        + _fmt(report.false_alarms_per_hour, ".2f")
        + "**  (target < 0.125, i.e. under one per 8-hour shift)"
    )
    add("")
    add("| metric | value |")
    add("|---|---|")
    add("| recall | " + _fmt(report.recall) + " |")
    add("| precision | " + _fmt(report.precision) + " |")
    add("| F1 | " + _fmt(report.f1) + " |")
    add("| falls detected | " + str(report.n_tp) + " |")
    add("| falls missed | " + str(report.n_miss) + " |")
    add("| false alarms | " + str(report.n_fp) + " |")
    add(
        "| alert latency (median / p90) | "
        + _fmt(report.latency_median(), ".1f")
        + " s / "
        + _fmt(report.latency_p90(), ".1f")
        + " s |"
    )
    add("")

    misses = [(r.clip_id, t) for r in report.results for t in r.misses]
    if misses:
        add("## Missed falls")
        add("")
        add("| clip | impact time |")
        add("|---|---|")
        for clip_id, t in misses:
            add("| " + clip_id + " | " + _fmt(t, ".1f") + " s |")
        add("")

    fps = [
        (r.clip_id, e) for r in report.results for e in r.false_positives
    ]
    if fps:
        add("## False alarms")
        add("")
        add("This table, not the F1, drives the next tuning pass.")
        add("")
        add("| clip | type | t_alert | zone | evidence |")
        add("|---|---|---|---|---|")
        for clip_id, e in fps:
            evidence = ", ".join(
                k + "=" + str(v) for k, v in sorted(e.evidence.items())
            )
            add(
                "| "
                + clip_id
                + " | "
                + e.type
                + " | "
                + _fmt(e.t_alert, ".1f")
                + " s | "
                + (e.zone or "-")
                + " | "
                + evidence
                + " |"
            )
        add("")

    add("## Per-clip")
    add("")
    add("| clip | hours | detected | missed | false alarms |")
    add("|---|---|---|---|---|")
    for r in report.results:
        add(
            "| "
            + r.clip_id
            + " | "
            + _fmt(r.duration_s / 3600.0, ".3f")
            + " | "
            + str(r.n_tp)
            + " | "
            + str(r.n_miss)
            + " | "
            + str(r.n_fp)
            + " |"
        )
    add("")

    return "\n".join(lines)
