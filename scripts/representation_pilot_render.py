"""Native Gaussian rasterizer audit and a preselected synthetic forecast demo."""
import numpy as np
import torch
from scipy.optimize import minimize_scalar

from wmfracture.rendering import build_renderer
from wmfracture.representation_pilot import interpolation
from spatial_observation_evaluate import write


def cameras(width):
    return [build_renderer({'width':width,'camera_eye':eye,'camera_target':[.5,0.,.01],
        'vertical_fov_degrees':45,'background':[0.,0.,0.]}) for eye in
        ([.5,-1.2,.85],[1.35,-1.0,.7],[-.35,-1.0,.7])]


def primitives(layout, x, z):
    sample = interpolation(x,layout['x']); height = sample@z
    slope = sample@np.gradient(z,x)
    gx = np.repeat(layout['x'],4); gy = np.tile(np.array([-.045,-.015,.015,.045]),len(height))
    gz = np.repeat(height,4); slope = np.repeat(slope,4)
    sx2, sy2 = layout['sigma_x']**2, layout['sigma_y']**2
    covariance = np.column_stack((np.full(len(gx),sx2),np.zeros(len(gx)),sx2*slope,
        np.full(len(gx),sy2),np.zeros(len(gx)),1e-6+sx2*slope*slope))
    colors = np.column_stack((.2+.6*gx,np.full(len(gx),.6),.8-.3*gx))
    return [torch.as_tensor(v,dtype=torch.float32,device='cuda').contiguous() for v in
        (np.column_stack((gx,gy,gz)),covariance,colors,
         np.clip(np.repeat(layout['opacity'],4)*layout['opacity_scale'],1e-5,.99)[:,None])]


def raster(camera, packed, auxiliary=False):
    renderer, meta = camera; xyz,cov,color,opacity = packed
    if auxiliary:
        matrix = torch.as_tensor(meta.world_to_camera,dtype=xyz.dtype,device='cuda')
        depth = xyz@matrix[2,:3]+matrix[2,3]
        color = torch.stack((torch.ones_like(depth),depth,torch.zeros_like(depth)),dim=1)
    with torch.no_grad():
        result,_,_ = renderer(means3D=xyz,means2D=torch.zeros_like(xyz),colors_precomp=color,
            opacities=opacity,cov3D_precomp=cov)
    return result.detach().cpu().numpy().transpose(1,2,0)


def audit(layouts, beam, config, output):
    views = cameras(config['render_width']); zero = np.zeros(len(beam.x))
    references = [raster(c,primitives(layouts['uniform'],beam.x,zero)) for c in views]
    for name, layout in layouts.items():
        if name == 'uniform': continue
        def objective(scale):
            layout['opacity_scale'] = float(scale)
            packed = primitives(layout,beam.x,zero)
            return sum(float(np.mean((raster(c,packed)-reference)**2)) for c,reference in zip(views,references))
        fit = minimize_scalar(objective,bounds=config['opacity_calibration_bounds'],method='bounded',
            options={'maxiter':config['opacity_calibration_iterations']})
        layout['opacity_scale'] = float(fit.x)
    poses = [zero,.035*np.sin(np.pi*beam.x)**2,.02*np.sin(np.pi*beam.x)**2*np.sin(2*np.pi*beam.x)]
    rows = []; query = np.linspace(0,1,129)
    for pi,z in enumerate(poses):
        z = z.copy(); z[[0,1,-2,-1]] = 0.
        reference = primitives(layouts['uniform'],beam.x,z)
        true_surface = interpolation(beam.x,query)@z
        reference_rgb = [raster(c,reference) for c in views]
        reference_aux = [raster(c,reference,True) for c in views]
        for name,layout in layouts.items():
            packed = primitives(layout,beam.x,z)
            sample = interpolation(beam.x,layout['x'])@z
            surface = interpolation(layout['x'],query)@sample
            surface_error = float(np.sqrt(np.mean((surface-true_surface)**2)))
            for vi,camera in enumerate(views):
                rgb,aux = raster(camera,packed),raster(camera,packed,True)
                ra,ga = reference_aux[vi][...,0],aux[...,0]
                rm,gm = ra>.05,ga>.05; union = rm|gm; common = rm&gm
                iou = float((rm&gm).sum()/max(union.sum(),1))
                rgb_error = float(np.sqrt(np.mean((rgb[union]-reference_rgb[vi][union])**2)))
                rd = reference_aux[vi][...,1]/np.maximum(ra,1e-8)
                gd = aux[...,1]/np.maximum(ga,1e-8)
                depth_error = float(np.sqrt(np.mean((rd[common]-gd[common])**2))) if common.any() else None
                rows.append({'representation':name,'pose':pi,'view':vi,'rgb_foreground_rmse':rgb_error,
                    'silhouette_iou':iou,'depth_rmse_m':depth_error,'surface_rmse_m':surface_error})
    summary = []
    for name in layouts:
        group = [r for r in rows if r['representation']==name]
        rgb = max(r['rgb_foreground_rmse'] for r in group); iou = min(r['silhouette_iou'] for r in group)
        depth = max(r['depth_rmse_m'] for r in group) if all(r['depth_rmse_m'] is not None for r in group) else None
        surface = max(r['surface_rmse_m'] for r in group)
        summary.append({'representation':name,'gaussians':len(layouts[name]['x'])*4,
            'opacity_scale_fit_on_rest_only':layouts[name]['opacity_scale'],
            'maximum_rgb_foreground_rmse':rgb,'minimum_silhouette_iou':iou,
            'maximum_depth_rmse_m':depth,'maximum_surface_rmse_m':surface,
            'admissible':bool(rgb<=config['audit_foreground_rgb_rmse'] and iou>=config['audit_silhouette_iou']
                and depth is not None and depth<=config['audit_depth_rmse'] and surface<=config['audit_surface_rmse'])})
    write(output/'appearance-audit.json',{'summary':summary,'rows':rows,
        'scope':'Procedural native alpha-composited 3D Gaussians. Scalar opacity calibrated on rest images only; analytic pose audit is not model training or a general reconstruction-equivalence proof.'})
    return summary


def demo(layout, beam, histories, output, width):
    import cv2
    view = cameras(width)[0]; columns = []
    for name, states in histories.items():
        images = []
        for state in states:
            z = beam.full(state)[:,0]; packed = primitives(layout,beam.x,z)
            rgb = raster(view,packed)
            image = np.clip(rgb*255,0,255).astype(np.uint8)
            image = np.pad(image,((24,0),(0,0),(0,0)))
            cv2.putText(image,name,(4,16),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
            images.append(image)
        columns.append(images)
    frames = [np.concatenate(parts,axis=1) for parts in zip(*columns)]
    writer = cv2.VideoWriter(str(output/'forecast.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(frames[0].shape[1],frames[0].shape[0]))
    if not writer.isOpened(): raise RuntimeError('Video writer unavailable')
    for image in frames: writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    writer.release(); cv2.imwrite(str(output/'forecast-end.png'),cv2.cvtColor(frames[-1],cv2.COLOR_RGB2BGR))
