import copy
import numpy as np
import pytest
from ahfd.ml.pilot import CausalPilotFeatures, anticipation_target, current_features

POLY=[[0,0],[100,0],[100,100],[0,100]]

def person(track_id=1,shift=0):
    return {'track_id':track_id,'score':.9,'keypoints':[[20+i+shift,20+2*i] for i in range(17)],'scores':[.9]*17,'heights':[-99]*17}

def test_no_metric_height_or_identity_features():
    a=current_features(person(),[[.1,.2,5]]*17,POLY)
    p=person();p['heights']=[999]*17;p['track_id']=123
    assert a==current_features(p,[[.1,.2,5]]*17,POLY)
    assert not any(k.startswith('dh_') for k in a)

def test_future_cannot_change_existing_features():
    a=CausalPilotFeatures(POLY);b=CausalPilotFeatures(POLY)
    for i in range(15):
        assert a.update(i*.1,person(shift=i),[[0,0,4]]*17)==b.update(i*.1,person(shift=i),[[0,0,4]]*17)
    before=copy.deepcopy(a.history[0][1])
    a.update(1.5,person(shift=1000),[[0,0,9]]*17)
    assert before==b.history[0][1]

def test_gap_and_reassociation_reset():
    a=CausalPilotFeatures(POLY)
    for i in range(10):a.update(i*.1,person(),[[0,0,4]]*17)
    assert a.update(1.4,person(),[[0,0,4]]*17)['history_duration_s']==0
    assert a.update(1.5,person(2),[[0,0,4]]*17)['history_duration_s']==0
    assert a.update(1.6,None,None) is None

def test_missing_depth_is_not_zero_depth():
    row=current_features(person(),[None]*17,POLY)
    assert row['depth_shoulder_relative_hip_y'] is None
    assert row['depth_shoulder_relative_hip_y__missing']==1

def test_anticipation_censors_uncertainty_and_tail():
    assert anticipation_target(10,5,60,[(20,21)],'RECLINED')==0
    assert anticipation_target(17,5,60,[(20,21)],'UPRIGHT_IN_BED')==1
    assert anticipation_target(15.5,5,60,[(20,21)],'RECLINED') is None
    assert anticipation_target(20.5,5,60,[(20,21)],'UPRIGHT_IN_BED') is None
    assert anticipation_target(58,5,60,[],'RECLINED') is None
    assert anticipation_target(30,5,60,[(20,21)],'OUT_OF_BED') is None

def test_non_monotonic_rejected():
    a=CausalPilotFeatures(POLY);a.update(1,person(),[None]*17)
    with pytest.raises(ValueError):a.update(.5,person(),[None]*17)


def test_empty_pose_placeholders_are_unavailable_and_reset_history():
    a=CausalPilotFeatures(POLY)
    a.update(0,person(),[None]*17)
    empty={'track_id':None,'score':0,'keypoints':[[0,0]]*17,'scores':[0]*17}
    assert a.update(.1,empty,[None]*17) is None
    assert a.update(.2,person(),[None]*17)['history_duration_s']==0
    low=person();low['scores']=[0]*17
    assert a.update(.3,low,[None]*17) is None
