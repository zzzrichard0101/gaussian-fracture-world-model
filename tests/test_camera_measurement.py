import numpy as np
import pytest

from wmfracture.camera_measurement import summarize,track_reference


def test_known_subpixel_common_motion_does_not_become_a_fake_gap_signal():
    cv2 = pytest.importorskip('cv2')
    rng = np.random.default_rng(3017)
    reference = np.zeros((128,256),np.uint8)
    for x in (60,190):
        reference[35:85,x-25:x+25] = rng.integers(20,240,(50,50),dtype=np.uint8)
    initial = np.array([[60.,60.],[190.,60.]],np.float32)
    points,valid = [],[]
    for i in range(40):
        transform = np.array([[1.,0.,.8*np.sin(i*.3)],[0.,1.,.5*np.cos(i*.2)]],np.float32)
        current = cv2.warpAffine(reference,transform,(256,128))
        current = np.clip(current.astype(float)+rng.normal(0,1.5,current.shape),0,255).astype(np.uint8)
        tracked,ok = track_reference(reference,current,initial)
        points.append(tracked); valid.append(ok)
    report,gaps,mask = summarize(points,valid,np.arange(40)/30,.05,signal_mm=.069)
    assert report['valid_fraction'] >= .95
    assert np.max(np.abs(gaps[mask]-6.5)) < .01
    assert report['gap_repeatability_std_mm'] < .005
    assert not report['absolute_accuracy_established']
    assert report['separation_gate_status'].startswith('incomplete')


def test_repeatability_does_not_silently_prove_bias_or_average_away_frames():
    points = np.zeros((10,2,2)); points[:,1,0] = 100+np.linspace(-1,1,10)
    report,_,_ = summarize(points,np.ones(10,bool),np.arange(10),.05,signal_mm=.069,bias_bound_mm=.05)
    assert not report['meets_conditional_repeatability_budget']
    assert report['conditional_separation_requirement_mm'] >= .1
    assert report['gap_repeatability_std_mm'] == pytest.approx(np.std(np.linspace(-1,1,10)*.05,ddof=1))


def test_subpixel_relative_motion_is_measured_instead_of_returning_the_reference():
    cv2 = pytest.importorskip('cv2')
    rng = np.random.default_rng(7801)
    left,right = np.zeros((128,256),np.uint8),np.zeros((128,256),np.uint8)
    left[35:85,35:85] = rng.integers(20,240,(50,50),dtype=np.uint8)
    right[35:85,165:215] = rng.integers(20,240,(50,50),dtype=np.uint8)
    initial = np.array([[60.,60.],[190.,60.]],np.float32)
    common = np.array([[1.,0.,.3],[0.,1.,.2]],np.float32)
    moved = common.copy(); moved[0,2] += 1.4
    current = cv2.add(cv2.warpAffine(left,common,(256,128)),cv2.warpAffine(right,moved,(256,128)))
    tracked,valid = track_reference(cv2.add(left,right),current,initial)
    assert valid
    measured_change_mm = (tracked[1,0]-tracked[0,0]-130)*.05
    assert measured_change_mm == pytest.approx(.07,abs=.0025)


def test_invalid_frames_are_counted_and_insufficient_data_is_explicit():
    points = np.zeros((5,2,2)); points[:,1,0] = 100
    points[1] = np.nan
    result,_,mask = summarize(points,[True]*5,np.arange(5),.05)
    assert result['valid_frames'] == 4 and not mask[1]
    with pytest.raises(ValueError):
        summarize(points,[True,False,False,False,True],np.arange(5),.05)
