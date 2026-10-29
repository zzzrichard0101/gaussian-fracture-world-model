"""Scientific summaries from frozen pilot predictions, without tuning models."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from spatial_observation_evaluate import write
from wmfracture.representation_pilot import Beam, gaussian_layout, encoder_matrix


def main():
    project = Path(__file__).resolve().parents[1]; output = project/'artifacts/representation-pilot'
    report = json.loads((output/'report.json').read_text()); config = report['config']
    assert hashlib.sha256((output/'predictions.npz').read_bytes()).hexdigest() == report['frozen_predictions_sha256']
    sources = json.loads((output/'code-sha256.json').read_text())
    assert all(hashlib.sha256((project/p).read_bytes()).hexdigest() == digest for p,digest in sources.items())
    inputs = np.load(output/'public-inputs.npz')
    write(output/'inputs.json',{'initial_states':inputs['initial_states'].tolist(),
        'planned_force_accelerations':inputs['future_forces'].tolist(),
        'scope':'Known-state inputs in fixed material coordinates; no target future positions. Rows are grouped by forecast start, then seen/novel action episodes.'})
    beam = Beam(config['physical_nodes'],config['bending'],config['damping'],config['dt'])
    full_initial = beam.full(inputs['initial_states']); geometry_rows = []
    for name,spec in config['representations'].items():
        matrix = encoder_matrix(gaussian_layout(spec),beam.x,beam.x,'grid_aug')
        reconstructed = np.einsum('ij,bjk->bik',matrix,full_initial)
        error = reconstructed[:,beam.free]-full_initial[:,beam.free]
        geometry_rows.append({'representation':name,
            'initial_position_reconstruction_rmse_mm':float(np.sqrt(np.mean(error[...,0]**2))*1000),
            'initial_velocity_reconstruction_rmse_mm_s':float(np.sqrt(np.mean(error[...,1]**2))*1000)})
    write(output/'initial-geometry.json',{'rows':geometry_rows,'scope':'Known-input geometric interpolation floor at 48 forecast starts, not independent physical episodes or future-state estimation accuracy.'})
    rows = json.loads((output/'evaluation-rows.json').read_text())
    heldout = [name for name in config['representations'] if name not in config['train_representation_names']]
    methods = config['methods']+['grid_ridge','grid_cv']; aggregate = []
    for group in ('seen_positions','novel_positions'):
        for method in methods:
            group_rows = [r for r in rows if r['action_group']==group and r['method']==method and r['representation'] in heldout]
            finite = all(r['available'] for r in group_rows)
            seed_values = []
            for seed in config['training_seeds']:
                selected = [r for r in group_rows if r['seed']==seed and r['available']]
                seed_values.append(float(np.sqrt(np.mean([r['position_mse_m2'] for r in selected]))*1000) if selected else None)
            aggregate.append({'method':method,'action_group':group,'representations':heldout,
                'physical_episodes':config['test_episodes_per_action_group'],'seed_representation_episode_rows':len(group_rows),
                'all_finite':finite,'rollout_rmse_mm':float(np.sqrt(np.mean([r['position_mse_m2'] for r in group_rows]))*1000) if finite else None,
                'disagreement_rmse_mm':float(np.sqrt(np.mean([r['representation_disagreement_mse_m2'] for r in group_rows]))*1000) if finite else None,
                'seed_rollout_rmse_mm':seed_values})
    paired = []
    for group in ('seen_positions','novel_positions'):
        group_rows = [r for r in rows if r['action_group']==group and r['representation'] in heldout]
        case_ids = sorted(set(r['case'] for r in group_rows))
        differences = []
        for case in case_ids:
            errors = {m:np.mean([r['position_mse_m2'] for r in group_rows if r['method']==m and r['case']==case])
                for m in ('weight_aug','kernel_consistency')}
            differences.append(float(errors['kernel_consistency']-errors['weight_aug']))
        paired.append({'action_group':group,'physical_episodes':len(case_ids),
            'kernel_consistency_lower_mse_cases':sum(d<0 for d in differences),'paired_mse_difference_m2':differences})
    diagnostic = {'heldout_representation_aggregate':aggregate,'paired_candidate_vs_weight':paired,
        'all_prediction_rows_finite':all(r['available'] for r in rows),
        'maximum_predicted_absolute_position_m':max(r['maximum_absolute_position_m'] for r in rows if r['available']),
        'equivalence_scope':'Three analytic pose fields and three fixed cameras; not a guarantee for arbitrary poses, independent real refits or general 3D assets.',
        'statistical_scope':'Rows reuse the same physical episodes across layouts, checkpoints and two forecast windows; no independence/significance claim.'}
    write(output/'diagnostics.json',diagnostic)
    write(output/'integrity.json',{'primary_code_hashes_match':True,'primary_predictions_freeze_matches':True,
        'raw_predictions_sha256':report['frozen_predictions_sha256'],
        'checkpoint_hashes_match':all(hashlib.sha256((output/f"{r['method']}-{r['seed']}.pt").read_bytes()).hexdigest()==r['checkpoint_sha256'] for r in report['training_records']),
        'raw_scope':'Full NPZ arrays and checkpoint weights remain local artifacts; public exports contain selected inputs, episode metrics, training records, audit and a preselected forecast.'})
    write(output/'report-source-sha256.json',{'scripts/report_representation_pilot.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    chosen = ['point','point_aug','density_aug','weight_aug','kernel_consistency','grid_aug','grid_ridge']
    labels = ['Point','Point + aug','KDE + aug','Weight + aug','Kernel + consistency','Grid + aug','Grid ridge']
    fig,axes = plt.subplots(1,2,figsize=(12,4.8)); x = np.arange(len(chosen)); width = .36
    for i,group in enumerate(('seen_positions','novel_positions')):
        selected = [next(r for r in aggregate if r['method']==m and r['action_group']==group) for m in chosen]
        for axis,key in zip(axes,('rollout_rmse_mm','disagreement_rmse_mm')):
            axis.bar(x+(i-.5)*width,[r[key] for r in selected],width,label=group)
    axes[0].set_ylabel('16-step position RMSE (mm)');axes[1].set_ylabel('Forecast disagreement versus uniform (mm)')
    for axis in axes:
        axis.set_xticks(x,labels,rotation=30,ha='right');axis.grid(axis='y',alpha=.2);axis.legend(fontsize=8)
    fig.suptitle('One known-state beam; 5 held-out Gaussian layouts; 2 training seeds; same 35,258-parameter MLP')
    fig.text(.5,.015,'Grid ridge is a smaller linear diagnostic. Shared material coordinates and procedural splats; not image-only state estimation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94));fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    # Curves show subpixel motions directly in physical units; native render stays unexaggerated.
    demo = json.loads((output/'demo-forecast.json').read_text())['histories']; xx = np.linspace(0,1,config['physical_nodes'])[2:-2]
    fig,axis = plt.subplots(figsize=(8,3.5))
    for name,history in demo.items():axis.plot(xx,np.asarray(history)[-1,:len(xx)]*1000,label=name)
    axis.set(xlabel='Material coordinate (m)',ylabel='Out-of-plane displacement (mm)',title='Preselected demo: final forecast centerline, actual displacement')
    axis.legend();axis.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'forecast-centerline.png',dpi=150);plt.close(fig)
    print(json.dumps({'aggregate':aggregate,'paired':paired,'all_finite':diagnostic['all_prediction_rows_finite']},indent=2))


if __name__ == '__main__': main()
