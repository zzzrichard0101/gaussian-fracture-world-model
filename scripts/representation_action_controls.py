"""Post-primary frozen-model action ablations; no retraining or test selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from representation_pilot_evaluate import Transition, rollout
from spatial_observation_evaluate import write
from wmfracture.representation_pilot import Beam, gaussian_layout, encoder_matrix, episodes


def main():
    project = Path(__file__).resolve().parents[1]; output = project/'artifacts/representation-pilot'
    config = json.loads((output/'config.json').read_text()); plan = json.loads((project/'config/representation-controls.json').read_text())
    beam = Beam(config['physical_nodes'],config['bending'],config['damping'],config['dt'])
    data = np.load(output/'public-inputs.npz'); initial,forces = data['initial_states'],data['future_forces']
    layout = gaussian_layout(config['representations'][plan['representation']])
    permutation = np.random.default_rng(plan['permutation_seed']).permutation(len(forces))
    assert not np.array_equal(permutation,np.arange(len(forces)))
    predictions = {}; weights = {}
    for method in plan['methods']:
        matrix = encoder_matrix(layout,beam.x,beam.x,method,config['anchor_width'],config['density_bandwidth'])
        for seed in config['training_seeds']:
            path = output/f'{method}-{seed}.pt'; weights[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            model = Transition(len(beam.free),config['hidden_width']).cuda()
            model.load_state_dict(torch.load(path,map_location='cuda')); model.eval()
            for control in plan['controls']:
                changed = np.zeros_like(forces) if control=='zero_actions' else forces[permutation]
                predictions[f'{method}__{seed}__{control}'] = rollout(beam,initial,changed,matrix,config,model=model)
    np.savez_compressed(output/'action-controls-predictions.npz',**predictions)
    frozen = hashlib.sha256((output/'action-controls-predictions.npz').read_bytes()).hexdigest()
    # Reconstruct test futures only after action-control predictions freeze.
    batches = [episodes(beam,config['test_episodes_per_action_group'],config['data_seeds'][2]+i,centers,config)
        for i,centers in enumerate((config['train_action_centers'],config['novel_action_centers']))]
    states = np.concatenate([b['states'] for b in batches]); cases = len(states)
    target = np.concatenate([states[:,s+1:s+config['rollout_horizon']+1] for s in config['rollout_starts']])
    actual = np.load(output/'predictions.npz'); rows = []; n = len(beam.free)
    for method in plan['methods']:
        for group_index,group in enumerate(('seen_positions','novel_positions')):
            ids = np.arange(group_index*config['test_episodes_per_action_group'],(group_index+1)*config['test_episodes_per_action_group'])
            chosen = np.concatenate([ids+i*cases for i in range(len(config['rollout_starts']))])
            for control in ['actual_actions']+plan['controls']:
                errors = []
                for seed in config['training_seeds']:
                    p = actual[f'{method}__{seed}__uniform'] if control=='actual_actions' else predictions[f'{method}__{seed}__{control}']
                    errors.append(np.mean((p[chosen,:,:n]-target[chosen,:,:n])**2))
                rows.append({'method':method,'action_group':group,'control':control,
                    'rollout_position_rmse_mm':float(np.sqrt(np.mean(errors))*1000)})
    write(output/'action-controls.json',{'plan':plan,'checkpoint_sha256':weights,'control_predictions_sha256':frozen,
        'permutation':permutation.tolist(),'rows':rows,
        'source_sha256':{p:hashlib.sha256((project/p).read_bytes()).hexdigest() for p in
            ('config/representation-controls.json','scripts/representation_action_controls.py')}})
    print(json.dumps(rows,indent=2))


if __name__ == '__main__': main()
