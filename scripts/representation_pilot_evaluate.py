"""Small learned Gaussian-representation pilot, with fixed physics and held-out episodes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from wmfracture.representation_pilot import (Beam, gaussian_layout, encoder_matrix, episodes,
    normalized_features, normalized_state, decode_state)
from representation_pilot_render import audit, demo
from spatial_observation_evaluate import write


class Transition(nn.Module):
    def __init__(self, n, width):
        super().__init__(); self.n = n
        self.layers = nn.Sequential(nn.Linear(3*n,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,2*n))
        nn.init.zeros_(self.layers[-1].weight); nn.init.zeros_(self.layers[-1].bias)

    def forward(self, value):
        return value[:,:2*self.n]+self.layers(value)


def train(beam, train_data, validation, matrices, method, seed, config, output):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = Transition(len(beam.free),config['hidden_width']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    names = ['uniform'] if method == 'point' else config['train_representation_names']
    x = {name:torch.tensor(normalized_features(beam,beam.full(train_data['states'][:,:-1]),train_data['forces'],
            matrices[(method,name)],config).reshape(-1,3*len(beam.free)),dtype=torch.float32,device='cuda') for name in names}
    y = torch.tensor(normalized_state(beam,beam.full(train_data['states'][:,1:]),config).reshape(-1,2*len(beam.free)),dtype=torch.float32,device='cuda')
    validation_x = [torch.tensor(normalized_features(beam,beam.full(validation['states'][:,:-1]),validation['forces'],
        matrices[(method,name)],config).reshape(-1,3*len(beam.free)),dtype=torch.float32,device='cuda') for name in config['train_representation_names']]
    validation_y = torch.tensor(normalized_state(beam,beam.full(validation['states'][:,1:]),config).reshape(-1,2*len(beam.free)),dtype=torch.float32,device='cuda')
    rng = np.random.default_rng(seed+1); curves = []
    model.train()
    for step in range(1,config['steps']+1):
        index = torch.as_tensor(rng.integers(0,len(y),config['batch_size']),device='cuda')
        first,second = [names[int(i)] for i in rng.integers(0,len(names),2)]
        p,q = model(x[first][index]),model(x[second][index]); target = y[index]
        supervised = .5*((p-target).square().mean()+(q-target).square().mean())
        consistency = (p-q).square().mean()
        loss = supervised+(config['consistency_weight']*consistency if method=='kernel_consistency' else 0.)
        optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),10.); optimizer.step()
        if step % config['validation_every'] == 0:
            with torch.no_grad(): value = float(torch.stack([(model(v)-validation_y).square().mean() for v in validation_x]).mean())
            curves.append({'step':step,'training_supervised':float(supervised.detach()),'pair_consistency':float(consistency.detach()),'validation_mse':value})
    torch.cuda.synchronize(); elapsed = time.perf_counter()-start
    path = output/f'{method}-{seed}.pt'; torch.save(model.state_dict(),path)
    record = {'method':method,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),
        'seconds':elapsed,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'curves':curves,
        'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selection':'Final fixed-step checkpoint; no test-based or best-validation selection.'}
    return model.eval(),record


def rollout(beam, initial, forces, matrix, config, model=None, ridge=None, cv=False):
    state = initial.copy(); result = []
    if cv:
        reconstructed = np.einsum('ij,bjk->bik',matrix,beam.full(state))[:,beam.free]
        state = np.concatenate((reconstructed[...,0],reconstructed[...,1]),axis=1)
    for step in range(forces.shape[1]):
        if cv:
            n = len(beam.free); state = np.concatenate((state[:,:n]+beam.dt*state[:,n:],state[:,n:]),axis=1)
        else:
            features = normalized_features(beam,beam.full(state),forces[:,step],matrix,config)
            if model is not None:
                with torch.no_grad(): value = model(torch.tensor(features,dtype=torch.float32,device='cuda')).cpu().numpy()
            else:
                value = features[:,:2*len(beam.free)]+np.c_[features,np.ones(len(features))]@ridge
            state = decode_state(beam,value,config)
        result.append(state.copy())
    return np.stack(result,axis=1)


def main():
    project = Path(__file__).resolve().parents[1]; output = project/'artifacts/representation-pilot'
    output.mkdir(parents=True,exist_ok=True); config = json.loads((project/'config/representation-pilot.json').read_text())
    paths = ['config/representation-pilot.json','wmfracture/representation_pilot.py','scripts/representation_pilot_evaluate.py',
        'scripts/representation_pilot_render.py','tests/test_representation_pilot.py','wmfracture/rendering.py']
    write(output/'config.json',config);write(output/'code-sha256.json',{p:hashlib.sha256((project/p).read_bytes()).hexdigest() for p in paths})
    if not torch.cuda.is_available(): raise RuntimeError('This declared pilot requires the existing CUDA renderer and training runtime')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.perf_counter(); beam = Beam(config['physical_nodes'],config['bending'],config['damping'],config['dt'])
    layouts = {name:gaussian_layout(spec) for name,spec in config['representations'].items()}
    appearance = audit(layouts,beam,config,output)
    print('REPRESENTATION_AUDIT '+json.dumps(appearance),flush=True)
    training = episodes(beam,config['train_episodes'],config['data_seeds'][0],config['train_action_centers'],config)
    validation = episodes(beam,config['validation_episodes'],config['data_seeds'][1],config['train_action_centers'],config)
    tests = []; labels = []
    for gi,centers in enumerate((config['train_action_centers'],config['novel_action_centers'])):
        data = episodes(beam,config['test_episodes_per_action_group'],config['data_seeds'][2]+gi,centers,config)
        tests.append(data); labels.extend(['seen_positions' if gi==0 else 'novel_positions']*len(data['states']))
    test = {k:np.concatenate([d[k] for d in tests]) for k in tests[0]}
    initial = np.concatenate([test['states'][:,s] for s in config['rollout_starts']])
    future_forces = np.concatenate([test['forces'][:,s:s+config['rollout_horizon']] for s in config['rollout_starts']])
    np.savez_compressed(output/'public-inputs.npz',initial_states=initial,future_forces=future_forces)
    methods = config['methods']+['grid_ridge','grid_cv']
    matrices = {(method,name):encoder_matrix(layout,beam.x,beam.x,
        'grid_aug' if method in ('grid_ridge','grid_cv') else method,config['anchor_width'],config['density_bandwidth'])
        for method in methods for name,layout in layouts.items()}
    predictions = {}; records = []
    for seed in config['training_seeds']:
        for method in methods:
            model = None; ridge = None
            if method in config['methods']:
                model,record = train(beam,training,validation,matrices,method,seed,config,output); records.append(record)
                print(f'REPRESENTATION_TRAIN seed={seed} method={method} validation={record["curves"][-1]["validation_mse"]:.6g} seconds={record["seconds"]:.2f}',flush=True)
            elif method == 'grid_ridge':
                features = np.concatenate([normalized_features(beam,beam.full(training['states'][:,:-1]),training['forces'],matrices[(method,name)],config).reshape(-1,3*len(beam.free)) for name in config['train_representation_names']])
                target = np.tile(normalized_state(beam,beam.full(training['states'][:,1:]),config).reshape(-1,2*len(beam.free)),(len(config['train_representation_names']),1))
                design = np.c_[features,np.ones(len(features))]; residual = target-features[:,:2*len(beam.free)]
                regularizer = np.eye(design.shape[1])*1e-5;regularizer[-1,-1] = 0.
                ridge = np.linalg.solve(design.T@design+regularizer,design.T@residual)
                np.save(output/f'ridge-{seed}.npy',ridge)
            for name in layouts:
                predictions[f'{method}__{seed}__{name}'] = rollout(beam,initial,future_forces,matrices[(method,name)],config,model,ridge,method=='grid_cv')
            if model is not None: del model; torch.cuda.empty_cache()
    np.savez_compressed(output/'predictions.npz',**predictions);write(output/'training.json',records)
    frozen = hashlib.sha256((output/'predictions.npz').read_bytes()).hexdigest()
    # Test futures are used for scoring only after prediction files are frozen.
    targets = np.concatenate([test['states'][:,s+1:s+config['rollout_horizon']+1] for s in config['rollout_starts']])
    cases = len(labels); n = len(beam.free); rows = []; summary = []
    admissible = {r['representation']:r['admissible'] for r in appearance}
    for key,pred in predictions.items():
        method,seed,name = key.split('__'); reference = predictions[f'{method}__{seed}__uniform']
        for ci in range(cases):
            chosen = [ci+i*cases for i in range(len(config['rollout_starts']))]
            p,t,r = pred[chosen],targets[chosen],reference[chosen]
            available = bool(np.isfinite(p).all())
            rows.append({'method':method,'seed':int(seed),'representation':name,'case':ci,'action_group':labels[ci],
                'admissible':admissible[name],'available':available,
                'position_mse_m2':float(np.mean((p[...,:n]-t[...,:n])**2)) if available else None,
                'velocity_mse_m2_s2':float(np.mean((p[...,n:]-t[...,n:])**2)) if available else None,
                'representation_disagreement_mse_m2':float(np.mean((p[...,:n]-r[...,:n])**2)) if available and np.isfinite(r).all() else None,
                'one_step_position_mse_m2':float(np.mean((p[:,0,:n]-t[:,0,:n])**2)) if available else None,
                'maximum_absolute_position_m':float(np.max(abs(p[...,:n]))) if available else None})
    for method in methods:
        for group in ('seen_positions','novel_positions'):
            for name in layouts:
                selected = [r for r in rows if r['method']==method and r['action_group']==group and r['representation']==name]
                valid = [r for r in selected if r['available']]
                summary.append({'method':method,'action_group':group,'representation':name,'admissible':admissible[name],
                    'physical_episodes':len(selected)//len(config['training_seeds']),'seed_episode_rows':len(selected),'available':len(valid),
                    'rollout_position_rmse_mm':float(np.sqrt(np.mean([r['position_mse_m2'] for r in valid]))*1000) if valid else None,
                    'rollout_velocity_rmse_mm_s':float(np.sqrt(np.mean([r['velocity_mse_m2_s2'] for r in valid]))*1000) if valid else None,
                    'representation_disagreement_rmse_mm':float(np.sqrt(np.mean([r['representation_disagreement_mse_m2'] for r in valid]))*1000) if valid else None,
                    'one_step_position_rmse_mm':float(np.sqrt(np.mean([r['one_step_position_mse_m2'] for r in valid]))*1000) if valid else None})
    write(output/'evaluation-rows.json',rows);write(output/'analysis-summary.json',summary)
    # First novel-position episode, first start, first seed, fixed right-dense layout: no quality-based selection.
    index = config['test_episodes_per_action_group']; seed = config['training_seeds'][0]
    histories = {'truth':np.r_[initial[index:index+1],targets[index]]}
    for method in ('point_aug','grid_aug','kernel_consistency'):
        histories[method] = np.r_[initial[index:index+1],predictions[f'{method}__{seed}__right_dense'][index]]
    write(output/'demo-forecast.json',{'selection':'First novel-position episode, first start and seed; right_dense layout fixed before scoring.',
        'histories':{k:v.tolist() for k,v in histories.items()}})
    demo(layouts['right_dense'],beam,histories,output,config['render_width'])
    report = {'config':config,'scope':config['scope'],'appearance_audit':appearance,
        'train_episodes':len(training['states']),'validation_episodes':len(validation['states']),'test_episodes':cases,
        'prediction_arrays':len(predictions),'evaluation_rows':len(rows),'frozen_predictions_sha256':frozen,
        'training_records':[{k:v for k,v in r.items() if k!='curves'} for r in records],
        'runtime':{'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'cuda':torch.version.cuda},
        'elapsed_seconds':time.perf_counter()-started,
        'memory_scope':'CUDA peak allocated is Torch allocator memory during each tiny cached-input training, not whole-device VRAM or a general image pipeline.',
        'encoding_scope':'Known material coordinates; fixed rest supports allow cached linear Gaussian-to-control matrices. Not an online learned image encoder; setup and raw Gaussian encoding are not included in training timings.'}
    write(output/'report.json',report);print(json.dumps({'elapsed_seconds':report['elapsed_seconds'],'evaluation_rows':len(rows),'frozen':frozen}),flush=True)


if __name__ == '__main__': main()
