"""Analyze an actual static video; never substitute synthetic camera precision."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path

import cv2
import numpy as np

from history_refinement import write_json,digest
from wmfracture.camera_measurement import track_reference,summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',required=True,type=Path)
    parser.add_argument('--points',nargs=4,type=float,metavar=('LEFT_X','LEFT_Y','RIGHT_X','RIGHT_Y'))
    parser.add_argument('--mm-per-pixel',type=float)
    parser.add_argument('--axis',nargs=2,type=float,default=[1.,0.])
    parser.add_argument('--signal-mm',type=float,default=.069)
    parser.add_argument('--bias-bound-mm',type=float)
    parser.add_argument('--max-frames',type=int,default=600)
    parser.add_argument('--extract-frame',action='store_true')
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error('Video file does not exist')
    if not args.extract_frame and (args.points is None or args.mm_per_pixel is None):
        parser.error('Analysis requires --points and --mm-per-pixel; first use --extract-frame if needed')
    root = Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-camera-repeatability')
    root.mkdir(parents=True)
    cap = cv2.VideoCapture(str(args.video))
    ok,first = cap.read()
    if not ok:
        cap.release(); parser.error('Could not decode the first frame')
    cv2.imwrite(str(root/'first-frame.png'),first)
    if args.extract_frame:
        cap.release(); print(f'FIRST_FRAME={root/"first-frame.png"}',flush=True); return
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0 or args.max_frames < 3:
        cap.release(); parser.error('Video needs a valid frame rate and at least three frames')
    reference = cv2.cvtColor(first,cv2.COLOR_BGR2GRAY)
    initial = np.array(args.points,dtype=np.float32).reshape(2,2)
    positions,valid,times = [initial],[True],[0.]
    try:
        while len(positions) < args.max_frames:
            ok,frame = cap.read()
            if not ok:
                break
            if frame.shape != first.shape:
                raise ValueError('Frame dimensions changed')
            points,status = track_reference(reference,cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),initial)
            times.append(len(positions)/fps); positions.append(points); valid.append(status)
    finally:
        cap.release()
    report,gaps,mask = summarize(positions,valid,times,args.mm_per_pixel,args.axis,args.signal_mm,args.bias_bound_mm)
    report.update({'video':str(args.video),'video_sha256':digest(args.video),'fps_metadata':fps,
        'timing_assumption':'Constant frame rate from metadata; not a variable-frame-rate timing calibration',
        'points_px':initial.tolist(),'mm_per_pixel':args.mm_per_pixel,'axis':args.axis,'run_directory':str(root)})
    write_json(root/'report.json',report)
    np.savez_compressed(root/'trace.npz',points=np.array(positions),valid=mask,time=np.array(times),gap_mm=gaps)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(9,3.5),constrained_layout=True)
    ax.plot(np.array(times)[mask],gaps[mask]-report['median_marker_distance_mm'])
    ax.set(xlabel='Nominal time (s)',ylabel='Gap deviation (mm)',title='Static repeatability; absolute accuracy not established')
    ax.grid(alpha=.3); fig.savefig(root/'repeatability.png',dpi=150); plt.close(fig)
    print(f'OUTPUT={root}',flush=True)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
