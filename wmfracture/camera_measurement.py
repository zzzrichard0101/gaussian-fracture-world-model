"""Two-marker static repeatability, not absolute camera accuracy calibration."""
import numpy as np


def track_reference(reference,current,points,fb_limit_px=.5):
    import cv2
    points = np.asarray(points,np.float32)
    if reference.ndim != 2 or current.shape != reference.shape or points.shape != (2,2) or not np.isfinite(points).all():
        raise ValueError('Expected same-size gray images and two image coordinates')
    if (points < 0).any() or (points[:,0] >= reference.shape[1]).any() or (points[:,1] >= reference.shape[0]).any() or fb_limit_px <= 0:
        raise ValueError('Invalid tracking coordinates or threshold')
    # Suppress interpolation-scale texture aliasing in both directions; retain
    # the forward/backward rejection rather than loosening it on bad tracks.
    reference = cv2.GaussianBlur(reference,(5,5),1.)
    current = cv2.GaussianBlur(current,(5,5),1.)
    original = points.reshape(2,1,2)
    options = dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.001))
    tracked,status,_ = cv2.calcOpticalFlowPyrLK(reference,current,original,None,**options)
    if tracked is None or status is None or not status.all():
        return np.full((2,2),np.nan),False
    back,reverse_status,_ = cv2.calcOpticalFlowPyrLK(current,reference,tracked,None,**options)
    valid = back is not None and reverse_status is not None and reverse_status.all() and np.isfinite(tracked).all()
    valid = bool(valid and np.max(np.linalg.norm(back-original,axis=2)) <= fb_limit_px)
    return tracked.reshape(2,2),valid


def summarize(points,valid,times,mm_per_pixel,axis=(1.,0.),signal_mm=None,bias_bound_mm=None):
    points,times,valid = np.asarray(points,float),np.asarray(times,float),np.asarray(valid,bool)
    axis = np.asarray(axis,float)
    if (points.ndim != 3 or points.shape[1:] != (2,2) or len(times) != len(points) or valid.shape != (len(points),)
        or not np.isfinite(mm_per_pixel) or mm_per_pixel <= 0 or axis.shape != (2,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 0):
        raise ValueError('Invalid measurement arrays, scale or axis')
    axis = axis/np.linalg.norm(axis)
    valid = valid & np.isfinite(points).all(axis=(1,2)) & np.isfinite(times)
    if valid.sum() < 3 or (np.diff(times[valid]) <= 0).any():
        raise ValueError('At least three valid, increasing measurement times are required')
    gaps = ((points[:,1]-points[:,0]) @ axis)*mm_per_pixel
    selected = gaps[valid]
    median = float(np.median(selected))
    sigma = float(np.std(selected,ddof=1))
    mad_sigma = float(1.4826*np.median(np.abs(selected-median)))
    drift = float(np.polyfit(times[valid]-times[valid][0],selected-median,1)[0])
    result = {'frames':len(points),'valid_frames':int(valid.sum()),'valid_fraction':float(valid.mean()),
        'median_marker_distance_mm':median,'gap_repeatability_std_mm':sigma,'gap_repeatability_mad_sigma_mm':mad_sigma,
        'gap_5_to_95_span_mm':float(np.quantile(selected,.95)-np.quantile(selected,.05)),
        'linear_drift_mm_per_s':drift,'absolute_accuracy_established':False,
        'scope':'Static clip repeatability only. Does not measure bias, scale accuracy, real material damage or dynamic tracking accuracy.'}
    if signal_mm is not None:
        if not np.isfinite(signal_mm) or signal_mm <= 0:
            raise ValueError('Expected positive signal in millimeters')
        result['requested_signal_mm'] = signal_mm
        result['two_independent_observation_3sigma_requirement_mm'] = float(3*np.sqrt(2)*sigma)
        if bias_bound_mm is not None:
            if not np.isfinite(bias_bound_mm) or bias_bound_mm < 0:
                raise ValueError('Invalid externally supplied bias bound')
            result['externally_supplied_per_observation_bias_bound_mm'] = bias_bound_mm
            result['conditional_separation_requirement_mm'] = float(2*bias_bound_mm+3*np.sqrt(2)*sigma)
            result['meets_conditional_repeatability_budget'] = bool(valid.mean() >= .9 and signal_mm > result['conditional_separation_requirement_mm'])
        else:
            result['separation_gate_status'] = 'incomplete_without_independent_bias_and_scale_assessment'
    return result,gaps,valid
