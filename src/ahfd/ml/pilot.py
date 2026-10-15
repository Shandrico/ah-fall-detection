"""Causal features for an OFFLINE, fixed-view, single-participant pilot.

This is not the governed onsite dataset loader, nor a live alert model. Image
bed distances are dimensionless projection features, NOT metric distances or
proof of bed support. Provisional floor heights are deliberately never read.
The caller supplies separately reviewed labels and explicit participant IDs.
"""
from __future__ import annotations

from collections import deque
import math
import cv2
import numpy as np


def _mid(points, valid, ids):
    chosen = [points[i] for i in ids if valid[i]]
    return np.mean(chosen, axis=0) if chosen else None


def _angle(points, valid, chain):
    if not all(valid[i] for i in chain):
        return None
    a, b, c = (points[i] for i in chain)
    x, y = a-b, c-b
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    if denominator < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(x, y)/denominator, -1, 1))))


def _mean_present(values):
    present = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(present)) if present else None


def current_features(person, xyz, polygon, min_score=.45):
    """No filenames, bed IDs, clip clock, script labels or floor heights."""
    points = np.asarray(person['keypoints'], dtype=float)
    scores = np.asarray(person['scores'], dtype=float)
    if points.shape != (17, 2) or scores.shape != (17,):
        raise ValueError('Expected COCO-17 pose')
    valid = (scores >= min_score) & np.isfinite(points).all(axis=1)
    polygon = np.asarray(polygon, np.float32)
    scale = math.sqrt(abs(cv2.contourArea(polygon)))
    if scale < 1:
        raise ValueError('Degenerate bed outline')
    centre = polygon.mean(axis=0)
    shoulder = _mid(points, valid, (5, 6))
    hip = _mid(points, valid, (11, 12))
    row = {'pose_confidence': float(person['score']), 'joint_visibility': float(valid.mean())}
    for name, mid in [('shoulder', shoulder), ('hip', hip), ('ankle', _mid(points, valid, (15, 16)))]:
        row[name+'_bed_x'] = float((mid[0]-centre[0])/scale) if mid is not None else None
        row[name+'_bed_y'] = float((mid[1]-centre[1])/scale) if mid is not None else None
        row[name+'_image_edge_distance'] = float(cv2.pointPolygonTest(polygon, tuple(map(float,mid)), True)/scale) if mid is not None else None
    if shoulder is not None and hip is not None and np.linalg.norm(shoulder-hip)>1:
        delta = shoulder-hip
        row['torso_tilt'] = float(np.degrees(np.arccos(np.clip(-delta[1]/np.linalg.norm(delta),-1,1))))
        row['torso_length_norm'] = float(np.linalg.norm(delta)/scale)
    else:
        row['torso_tilt'] = row['torso_length_norm'] = None
    visible = points[valid]
    row['bbox_aspect'] = float(np.ptp(visible[:,1])/max(np.ptp(visible[:,0]),1)) if len(visible)>1 else None
    row['knee_angle'] = _mean_present([_angle(points,valid,c) for c in [(11,13,15),(12,14,16)]])
    row['hip_angle'] = _mean_present([_angle(points,valid,c) for c in [(5,11,13),(6,12,14)]])
    camera = np.asarray([v if v is not None else [np.nan]*3 for v in xyz], dtype=float)
    if camera.shape != (17,3):
        raise ValueError('Expected 17 sparse camera XYZ samples')
    depth_valid = valid & np.isfinite(camera).all(axis=1) & (camera[:,2]>=.3) & (camera[:,2]<=10)
    row['depth_visibility'] = float(depth_valid.mean())
    camera_hip = _mid(camera,depth_valid,(11,12))
    for name, ids in [('shoulder',(5,6)),('head',(0,1,2)),('knee',(13,14)),('ankle',(15,16))]:
        mid = _mid(camera,depth_valid,ids)
        for j,axis in enumerate(('x','y','z')):
            row['depth_'+name+'_relative_hip_'+axis] = float(mid[j]-camera_hip[j]) if mid is not None and camera_hip is not None else None
    for name,value in list(row.items()):
        row[name+'__missing'] = float(value is None)
    return row


class CausalPilotFeatures:
    """Trailing history only; reset on reassociation or a >0.3 second gap."""
    def __init__(self, polygon):
        self.polygon = polygon
        self.history = deque()
        self.track_id = None

    def update(self, t, person, xyz):
        if not math.isfinite(t):
            raise ValueError('Invalid timestamp')
        if self.history and t<=self.history[-1][0]:
            raise ValueError('Timestamps must increase')
        if (person is None or person.get('track_id') is None
                or not math.isfinite(float(person.get('score', 0)))
                or person.get('score', 0) <= 0
                or not any(s >= .45 for s in person.get('scores', []))):
            self.history.clear();self.track_id=None
            return None
        track_id = person.get('track_id')
        if track_id != self.track_id or (self.history and t-self.history[-1][0]>.3):
            self.history.clear()
        self.track_id = track_id
        now = current_features(person,xyz,self.polygon)
        self.history.append((t,now))
        while self.history and t-self.history[0][0]>2.0+1e-6:
            self.history.popleft()
        result = dict(now)
        result['history_duration_s'] = t-self.history[0][0]
        keys = ('shoulder_bed_y','hip_bed_x','hip_bed_y','hip_image_edge_distance','torso_tilt','depth_shoulder_relative_hip_y','depth_shoulder_relative_hip_z')
        for window in (.5,2.):
            history = [(ts,row) for ts,row in self.history if t-ts<=window+1e-6]
            for key in keys:
                values = [(ts,row[key]) for ts,row in history if row[key] is not None]
                prefix = f'{key}__{window:g}s_'
                enough = len(values)>=2 and values[-1][0]-values[0][0]>=window*.5 and now[key] is not None
                result[prefix+'delta'] = values[-1][1]-values[0][1] if enough else None
                result[prefix+'range'] = max(v for _,v in values)-min(v for _,v in values) if enough else None
                result[prefix+'missing'] = float(not enough)
        return result


def anticipation_target(t, horizon, duration, exit_intervals, current_phase):
    """Censor uncertain event boundaries and the unobserved end of the clip.

    An interval is [last definitely bed-supported, first definitely exited].
    No future information enters features; future event times are targets only.
    """
    if horizon<=0 or t<1 or t+horizon>duration or current_phase in (None,'OUT_OF_BED'):
        return None
    for lo,hi in exit_intervals:
        if not (0<=lo<hi<=duration):
            raise ValueError('Invalid exit uncertainty interval')
        if lo<=t<=hi or lo<t+horizon<hi:
            return None
        if t<lo and hi<=t+horizon:
            return 1
    return 0
