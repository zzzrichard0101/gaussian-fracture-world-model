"""Post-primary converged linear optimizer control for the deliberately linear beam."""
import hashlib
import json
from pathlib import Path

import numpy as np

from representation_pilot_evaluate import rollout
from spatial_observation_evaluate import write
from wmfracture.representation_pilot import (Beam, gaussian_layout, encoder_matrix, episodes,
    normalized_features, normalized_state)


def main():
    project = Path(__file__).resolve().parents[1]; output = project/'artifacts/representation-pilot'
    config = json.loads((output/'config.json').read_text())
    beam = Beam(config['physical_nodes'],config['bending'],config['damping'],config['dt']); n = len(beam.free)
    layouts = {name:gaussian_layout(spec) for name,spec in config['representations'].items()}
    data = episodes(beam,config['train_episodes'],config['data_seeds'][0],config['train_action_centers'],config)
    target = normalized_state(beam,beam.full(data['states'][:,1:]),config).reshape(-1,2*n)
    inputs = np.load(output/'public-inputs.npz'); initial,forces = inputs['initial_states'],inputs['future_forces']
    heldout = [name for name in layouts if name not in config['train_representation_names']]
    predictions = {}; weights = {}
    for method in config['methods']:
        matrices = {name:encoder_matrix(layout,beam.x,beam.x,method,config['anchor_width'],config['density_bandwidth']) for name,layout in layouts.items()}
        names = ['uniform'] if method=='point' else config['train_representation_names']
        features = [normalized_features(beam,beam.full(data['states'][:,:-1]),data['forces'],matrices[name],config).reshape(-1,3*n) for name in names]
        designs = [np.c_[x,np.ones(len(x))] for x in features]
        design = np.concatenate(designs); residual = np.concatenate([target-x[:,:2*n] for x in features])
        regularizer = np.eye(design.shape[1])*1e-5; regularizer[-1,-1] = 0.
        lhs = design.T@design+regularizer; rhs = design.T@residual
        if method=='kernel_consistency':
            delta = designs[0]-designs[1]; base_delta = features[0][:,:2*n]-features[1][:,:2*n]
            # Two supervised branches and the paired full-prediction consistency objective.
            factor = len(names)*config['consistency_weight']
            lhs += factor*(delta.T@delta); rhs -= factor*(delta.T@base_delta)
        fitted = np.linalg.solve(lhs,rhs); weights[method] = fitted
        for name in layouts: predictions[f'{method}__{name}'] = rollout(beam,initial,forces,matrices[name],config,ridge=fitted)
    np.savez_compressed(output/'linear-control-predictions.npz',**predictions)
    frozen = hashlib.sha256((output/'linear-control-predictions.npz').read_bytes()).hexdigest()
    # Same exposed test cohort, scored only after the supplementary predictions freeze.
    batches = [episodes(beam,config['test_episodes_per_action_group'],config['data_seeds'][2]+i,centers,config)
        for i,centers in enumerate((config['train_action_centers'],config['novel_action_centers']))]
    states = np.concatenate([b['states'] for b in batches]); cases = len(states)
    target = np.concatenate([states[:,s+1:s+config['rollout_horizon']+1] for s in config['rollout_starts']])
    rows = []
    for method in config['methods']:
        for gi,group in enumerate(('seen_positions','novel_positions')):
            ids = np.arange(gi*config['test_episodes_per_action_group'],(gi+1)*config['test_episodes_per_action_group'])
            chosen = np.concatenate([ids+i*cases for i in range(len(config['rollout_starts']))])
            errors = []; mismatches = []
            for name in heldout:
                value = predictions[f'{method}__{name}'][chosen]
                errors.append(np.mean((value[...,:n]-target[chosen,:,:n])**2))
                mismatches.append(np.mean((value[...,:n]-predictions[f'{method}__uniform'][chosen,:,:n])**2))
            rows.append({'method':method,'action_group':group,'rollout_rmse_mm':float(np.sqrt(np.mean(errors))*1000),
                'disagreement_rmse_mm':float(np.sqrt(np.mean(mismatches))*1000)})
    original = np.load(output/'predictions.npz')
    replay_error = max(float(np.max(abs(predictions[f'grid_aug__{name}']-original[f'grid_ridge__{config["training_seeds"][0]}__{name}']))) for name in layouts)
    assert replay_error < 1e-10
    write(output/'linear-controls.json',{'scope':'Post-primary optimizer sanity control because fixed-step neural validation losses were still decreasing. Same exposed test cohort; no additional holdout, no retuning of primary models. Ridge=1e-5 as in the pre-existing grid_ridge diagnostic; exact quadratic solve, same encoders, paired full-prediction consistency for its ablation.',
        'rows':rows,'control_predictions_sha256':frozen,'original_grid_ridge_replay_max_abs':replay_error,
        'source_sha256':{'scripts/representation_linear_controls.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':main()
