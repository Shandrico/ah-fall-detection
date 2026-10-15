"""Causal warning simulation over saved pilot predictions, NOT live inference.

The scores are from models fitted on these same recordings. Patient/bed
identity is supplied by clip metadata, not evaluated here. Reference labels
are used only for the separate display comparison, never to produce warnings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ReplayPolicy:
    entry: float = .70
    release: float = .35
    persistence_s: float = .5
    release_s: float = .5
    cooldown_s: float = 3.
    max_gap_s: float = .3

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Policy values must be finite')
        if not 0 <= self.release < self.entry <= 1:
            raise ValueError('Require 0 <= release < entry <= 1')
        if min(self.persistence_s, self.release_s, self.cooldown_s) < 0 or self.max_gap_s <= 0:
            raise ValueError('Invalid elapsed-time policy')


def simulate(rows: list[dict], duration_s: float, policy: ReplayPolicy = ReplayPolicy()) -> dict:
    """Rows must be time-ordered. Gaps and reassociation break persistence.

    Each available sample supports coverage only until the next sample, capped
    at max_gap_s. No scoring during unavailable intervals and no interpolation.
    The caller must explicitly mark multi-person/unknown-association rows false.
    """
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError('Invalid duration')
    active = False
    candidate = low = last_track = previous = None
    next_allowed = 0.
    output, events = [], []
    coverage = 0.
    for i, row in enumerate(rows):
        t = float(row['t'])
        if not math.isfinite(t) or not 0 <= t <= duration_s or (previous is not None and t <= previous):
            raise ValueError('Invalid/non-increasing timestamp')
        score = row.get('score')
        track = row.get('track_id')
        available = bool(row.get('available')) and track is not None and score is not None
        available = available and math.isfinite(score) and 0 <= score <= 1 and row.get('phase') is not None
        if not available or track != last_track or (previous is not None and t - previous > policy.max_gap_s):
            active = False
            candidate = low = None
        previous = t
        last_track = track if available else None
        status = 'UNAVAILABLE'
        if available:
            end = rows[i+1]['t'] if i+1 < len(rows) else duration_s
            coverage += max(0., min(end-t, policy.max_gap_s, duration_s-t))
            in_bed_phase = row['phase'] in ('RECLINED', 'UPRIGHT_IN_BED', 'EDGE_SITTING')
            if active:
                if score <= policy.release or not in_bed_phase:
                    if low is None:
                        low = t
                    if t-low+1e-9 >= policy.release_s:
                        active = False
                        candidate = low = None
                else:
                    low = None
            elif score >= policy.entry and in_bed_phase:
                if candidate is None:
                    candidate = t
                if t-candidate+1e-9 >= policy.persistence_s and t+1e-9 >= next_allowed:
                    active = True
                    events.append({'t': t, 'candidate_start_s': candidate, 'score': score,
                                   'track_id': track, 'phase': row['phase'], 'type': 'SIMULATED_WARNING'})
                    next_allowed = t + policy.cooldown_s
                    candidate = None
            else:
                candidate = None
            status = 'WARNING' if active else 'BUILDING' if candidate is not None else 'NO_WARNING'
        output.append({**row, 'available': bool(available), 'status': status,
                       'score': score if available else None, 'phase': row.get('phase') if available else None})
    return {'rows': output, 'events': events, 'available_s': coverage,
            'unavailable_s': max(0., duration_s-coverage), 'policy': asdict(policy)}


def compare_reference(events: list[dict], exits: list[list[float]], horizon_s: float) -> dict:
    """Development display only, against coarse event uncertainty intervals.

    At most one warning is matched per reference exit. Remaining warnings stay
    visible (including duplicates); no false-alert/hour or clinical recall claim.
    A late window of 5 seconds is a displayed engineering choice, not a target.
    """
    if not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError('Invalid horizon')
    used = set()
    matches = []
    for lo, hi in sorted(exits):
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
            raise ValueError('Invalid reference interval')
        eligible = [(i,e) for i,e in enumerate(events) if i not in used and lo-horizon_s <= e['t'] <= hi+5]
        if not eligible:
            item = {'exit_interval_s': [lo,hi], 'classification': 'NO_MATCHING_WARNING'}
            earlier = [e for e in events if e['t'] < lo-horizon_s]
            if earlier:
                t = max(e['t'] for e in earlier)
                item.update(earlier_warning_s=t, earlier_lead_time_bounds_s=[lo-t,hi-t])
            matches.append(item)
            continue
        i, event = min(eligible, key=lambda pair: pair[1]['t'])
        used.add(i)
        t = event['t']
        category = 'EARLY' if hi-horizon_s <= t < lo else 'LATE' if t > hi else 'TIMING_UNCERTAIN'
        matches.append({'exit_interval_s': [lo,hi], 'warning_s': t,
                        'lead_time_bounds_s': [lo-t,hi-t], 'classification': category})
    return {'scope': 'coarse_training_clip_reference_only', 'matches': matches,
            'unmatched_warnings': [e for i,e in enumerate(events) if i not in used]}
