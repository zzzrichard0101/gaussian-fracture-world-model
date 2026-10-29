"""Fixed-protocol interval decisions on old witnesses and explicit controls."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from wmfracture.spatial_observation import PhysicalCatalog
from wmfracture.spatial_risk import verify, replay
from spatial_observation_evaluate import write


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--replay', type=Path)
    parser.add_argument('--followup-cells', type=int)
    args = parser.parse_args()
    if args.replay:
        proofs = json.loads(args.replay.read_text())
        for row in proofs: assert replay(row['proof'], row['query'])
        print(f'Replayed {len(proofs)} spatial interval/point decisions'); return
    project = Path(__file__).resolve().parents[1]; source = project/'docs/results'
    output = project/('artifacts/spatial-risk-refinement' if args.followup_cells else 'artifacts/spatial-risk'); output.mkdir(parents=True, exist_ok=True)
    config = json.loads((project/'config/spatial-continuous.json').read_text())
    settings = json.loads((source/'spatial-observation-config.json').read_text())
    witnesses = json.loads((source/'spatial-failure-search-witnesses.json').read_text())
    def query(prefix, observed, proposed, bounds=None):
        return {'parameter_bounds': config['parameter_bounds'] if bounds is None else bounds,
            'actions_m': [settings['actions_m'][i] for i in prefix], 'observed_m': observed,
            'positions_m': [-.04, .04], 'proposed_m': proposed, 'noise_radius_m': .00006,
            'bias_bound_m': .00015, 'damage_cap': .02, 'increments': config['cycle_increments']}
    inputs = []
    for i, old in enumerate(witnesses):
        row = old['input']; data = query(row['public_actions'], row['public_observed_m'], settings['actions_m'][row['action']])
        inputs.append({'id':f'old-public-witness-{i}', 'kind':'selected_old_failure', 'query':data,
            'warm_points':[old['witness']['parameter_point']], 'budget':1})
    point = [2., .0018, .0018]
    bank = PhysicalCatalog([point], settings['actions_m'], settings['future_drives_m'])
    observed = [bank.observations((0,), [-.04, .04])[0].tolist(), bank.observations((0,1), [-.04, .04])[0].tolist()]
    narrow = [[1.99999,2.00001],[.00179999,.00180001],[.00179999,.00180001]]
    for action in (0,2,4):
        inputs.append({'id':f'explicit-narrow-control-{action}', 'kind':'designed_local_control',
            'query':query([0,1], observed, settings['actions_m'][action], narrow), 'warm_points':[point], 'budget':1})
    wide = query([0,1], observed, settings['actions_m'][4])
    inputs.append({'id':'broad-domain-low-budget', 'kind':'broad_budget_control', 'query':wide, 'warm_points':[point], 'budget':config['risk_cell_budget']})
    zero = query([0,1], observed, [0.,0.])
    inputs.append({'id':'broad-domain-zero', 'kind':'broad_budget_control', 'query':zero, 'warm_points':[point], 'budget':config['risk_cell_budget']})
    impossible = query([0,1], [[.01,.01],[.01,.01]], [0.,0.])
    inputs.append({'id':'impossible-observations', 'kind':'empty_control', 'query':impossible, 'warm_points':[], 'budget':1})
    if args.followup_cells:
        inputs = [dict(row, budget=args.followup_cells) for row in inputs if row['kind'] == 'broad_budget_control']
        config = dict(config, followup_cells=args.followup_cells,
            followup_scope='Post-hoc budget sensitivity after observing 7-cell abstention. Original primary cases and results remain separate.')
    write(output/'inputs.json', inputs); write(output/'config.json', config)
    sources = ['wmfracture/spatial_intervals.py','wmfracture/spatial_risk.py','wmfracture/spatial_observation.py',
        'scripts/spatial_risk_evaluate.py','tests/test_spatial_intervals.py','tests/test_spatial_risk.py','config/spatial-continuous.json']
    write(output/'code-sha256.json', {p:hashlib.sha256((project/p).read_bytes()).hexdigest() for p in sources})
    results = []; summaries = []
    for row in inputs:
        tick = time.perf_counter(); proof = verify(row['query'], row['budget'], row['warm_points'])
        search_seconds = time.perf_counter()-tick
        tick = time.perf_counter(); assert replay(proof, row['query']); replay_seconds = time.perf_counter()-tick
        results.append({'id':row['id'], 'query':row['query'], 'proof':proof})
        leaves = [n for n in proof['nodes'] if n['children'] is None]
        summaries.append({'id':row['id'], 'kind':row['kind'], 'decision':proof['decision'],
            'cells':len(proof['nodes']), 'open_leaves':sum(n['status']=='open' for n in leaves),
            'max_unexcluded_upper':max((n['risk_upper'] for n in leaves if n['status']!='excluded'), default=None),
            'point_lower':max((p['risk_lower'] for p in proof['points'] if p['feasible']), default=None),
            'search_seconds':search_seconds,'replay_seconds':replay_seconds})
        print(f'SPATIAL_RISK {row["id"]} {proof["decision"]}', flush=True)
    write(output/'proofs.json', results); write(output/'analysis-summary.json', summaries)
    report = {'queries':len(inputs), 'decisions':{name:sum(r['decision']==name for r in summaries) for name in ('allow','reject','abstain','inconsistent')},
        'all_replayed':True,'summaries':summaries,
        'scope':'Conditional continuous-parameter coverage under the fixed 64-increment stable quasistatic real model. Narrow controls are designer-specified domains, not domains inferred from broad-prior observations. Broad-domain wrapping remains a failure of useful tightness. No closed-loop or real-world safety claim.'}
    write(output/'report.json',report); print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
