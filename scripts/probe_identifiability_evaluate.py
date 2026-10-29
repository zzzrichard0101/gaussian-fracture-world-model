"""Review diagnostic; no fitting, new policy, engine data or GPU required."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.continuous_risk import ContinuousRisk, outward, replay_proof
from wmfracture.probe_identifiability import PairProbeDiagnostic, gaussian_pair_score
from wmfracture.task_design import PublicRiskBank
from wmfracture.verified_policy import filter_proposal


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/probe-identifiability.json')
    parser.add_argument('--replay-proofs', type=Path)
    args = parser.parse_args()
    if args.replay_proofs:
        records = json.loads(args.replay_proofs.read_text())
        for row in records:
            assert replay_proof(row['proof'], expected_inputs=row['public_inputs'], expected_action=row['action_m'])
        print(f'Cold replayed {len(records)} published scalar proofs')
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    project = Path(__file__).resolve().parents[1]
    settings = json.loads((project/args.config).read_text())
    guard = json.loads((project/settings['guard_config']).read_text())
    law = json.loads((project/settings['law_config']).read_text())['law']
    output = project/'artifacts/probe-identifiability'; output.mkdir(parents=True, exist_ok=True)
    rows, sequences, filtered_rows, proofs = [], [], [], []
    for compliance in settings['matched_compliances']:
        for cap in settings['damage_caps']:
            inputs = {'parameter_bounds': guard['parameter_bounds'], 'spring': 10000.,
                'law': {k: law[k] for k in ('area', 'tensile_stiffness', 'peak_traction')},
                'actions_m': [0., guard['weak_drive_m']], 'observed_gaps_m': [0., 0.],
                'reference_count': 2, 'noise_radius_m': .00006,
                'bias_bound_m': guard['bias_bound_m'], 'damage_cap': cap}
            d = PairProbeDiagnostic(ContinuousRisk(inputs)); q = Q(compliance)
            pair = [(Q(e), d.matched_preparation(Q(e), q)) for e in settings['energies_j_m2']]
            result = d.maximum_difference(*pair, settings['maximum_drive_m'])
            best = result['best']; a = best['action']
            assert d.compliance(*pair[0]) == d.compliance(*pair[1]) == q
            assert d.model.damage(*pair[0]) == d.model.damage(*pair[1])
            # Every input up to the smaller preparation peak preserves both
            # histories and yields precisely the same gap, not only small SNR.
            nondamage_limit = min(h for _, h in pair)
            assert d.sequence_difference(*pair, [nondamage_limit, 0., nondamage_limit]).max() == 0.
            rows.append({'matched_compliance': compliance, 'damage_cap': cap,
                'parameters_exact': [[str(e), str(h)] for e, h in pair],
                'preparation_m': [float(h) for _, h in pair],
                'initial_damage': [float(d.model.damage(*p)) for p in pair],
                'pair_safe_ceiling_m': float(result['ceiling']),
                'max_difference_m': float(best['difference']), 'max_difference_exact_m': str(best['difference']),
                'maximizing_action_m': float(a), 'maximizing_action_exact_m': str(a),
                'damage_increments': [float(v) for v in best['damage_increments']],
                'breakpoint_count': len(result['candidates']),
                'future_gap_differences_m': [float(abs(d.model.gap(*pair[0], Q(u), Q(u))-d.model.gap(*pair[1], Q(u), Q(u)))) for u in settings['future_drives_m']],
                'scope': result['scope']})
            if cap != guard['damage_cap']: continue
            for sigma in settings['gaussian_sigmas_m']:
                # A designed noiseless public prefix with declared nonzero
                # uncertainty. q is used to GENERATE y, never passed to policy.
                public = deepcopy(inputs)
                public['observed_gaps_m'] = [0., float(q)*guard['weak_drive_m']]
                public['noise_radius_m'] = guard['noise_multiple']*sigma
                bank_guard = dict(guard, grid_side=settings['filter_grid_side'])
                bank = PublicRiskBank(public, bank_guard, InterfaceLaw(**law), Fixture())
                decision = filter_proposal(public, settings['maximum_drive_m'], bank.warm,
                    max_cells=settings['filter_cell_budget'], backtracking_steps=settings['filter_backtracking_steps'])
                selected, proof = decision['action_m'], decision['proof']
                assert selected is not None and proof['decision'] == 'allow' and replay_proof(proof)
                diff = abs(float(d.sequence_difference(*pair, [selected])[0]))
                filename = f'proof-q{compliance.replace("/", "_")}-s{sigma:g}.json.gz'
                with gzip.open(output/filename, 'wt', encoding='utf-8') as stream:
                    json.dump(proof, stream, allow_nan=False)
                # At or immediately BELOW the first informative branch switch.
                # A feasible violating witness here also excludes every larger
                # amplitude by monotonic cumulative damage in this model.
                boundary = outward(min(h for _, h in pair), False)
                threshold = ContinuousRisk(public).verify(boundary,
                    max_cells=settings['filter_cell_budget'], warm_points=bank.warm(boundary))
                assert replay_proof(threshold)
                threshold_name = 'threshold-'+filename
                with gzip.open(output/threshold_name, 'wt', encoding='utf-8') as stream:
                    json.dump(threshold, stream, allow_nan=False)
                for kind, value, evidence in (('selected', selected, proof), ('threshold', boundary, threshold)):
                    proofs.append({'matched_compliance': compliance, 'sigma_m': sigma, 'kind': kind,
                        'public_inputs': public, 'action_m': value, 'proof': evidence})
                filtered_rows.append({'matched_compliance': compliance, 'sigma_m': sigma,
                    'public_inputs': public, 'selected_action_m': selected, 'pair_difference_at_selected_m': diff,
                    'selected_proof_sha256': hashlib.sha256((output/filename).read_bytes()).hexdigest(),
                    'decision': proof['decision'], 'proof_replayed': True,
                    'first_information_threshold_lower_m': boundary,
                    'threshold_decision': threshold['decision'], 'threshold_proof_replayed': True,
                    'threshold_proof_sha256': hashlib.sha256((output/threshold_name).read_bytes()).hexdigest(),
                    'scope': 'One designed noiseless prefix; unchanged full-prior verifier and finite backtracking search. Selected action is not the largest safe continuous action; not an observed closed-loop campaign or full-domain information bound.'})
            for count in settings['observation_counts']:
                # Entire trace starts at the fixed reference state. Repeated
                # amplitudes do not reset D and do not cause cyclic fatigue in
                # this maximum-history law. Total count is matched per row.
                plans = {'repeat': [a]*count}
                if count >= 2:
                    plans['zero_probe_alternating'] = [Q(0) if i%2 == 0 else a for i in range(count)]
                for name, actions in plans.items():
                    delta = d.sequence_difference(*pair, actions)
                    assert np.max(abs(delta)) <= float(best['difference'])+1e-17
                    for sigma in settings['gaussian_sigmas_m']:
                        for tau in (0., settings['gaussian_bias_sigma_m']):
                            sequences.append({'matched_compliance': compliance, 'count': count, 'schedule': name,
                                'sigma_m': sigma, 'bias_sigma_m': tau,
                                **gaussian_pair_score(delta, sigma, tau)})
    # A 2-DOF rigid body's undamaged interface operator contains only three
    # weighted stiffness moments, regardless of camera pixel count.
    x = [Q(-1), Q(-1, 3), Q(1, 3), Q(1)]
    null = [Q(-1), Q(3), Q(-3), Q(1)]
    moments = [sum(n*p**order for n, p in zip(null, x)) for order in range(3)]
    assert moments == [0, 0, 0]
    rank_check = {'positions': [str(p) for p in x], 'stiffness_null_direction': [str(n) for n in null],
        'moment_changes_exact': [str(v) for v in moments],
        'scope': 'Four stiffness zones cannot in general be identified from a 2-DOF rigid pose response below damage onset: B^T diag(k) B has only three moments. A richer visual renderer does not add mechanical observables.'}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': settings,
        'pairs': rows, 'gaussian_schedules': sequences, 'filtered_prefix_diagnostics': filtered_rows,
        'spatial_rank_counterexample': rank_check,
        'method': 'Exact rational branch endpoints, not sampled input maxima. Gaussian scores use a distinct stated covariance model.',
        'scope': settings['scope']}
    write(output/'report.json', report); write(output/'config.json', settings)
    write(output/'proofs.json', proofs)
    paths = ['wmfracture/probe_identifiability.py', 'scripts/probe_identifiability_evaluate.py',
             'tests/test_probe_identifiability.py', 'wmfracture/continuous_risk.py',
             'wmfracture/task_design.py', 'wmfracture/verified_policy.py',
             args.config, settings['guard_config'], settings['law_config']]
    write(output/'code-sha256.json', {p: hashlib.sha256((project/p).read_bytes()).hexdigest() for p in paths})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for q in settings['matched_compliances']:
        subset = [r for r in rows if r['matched_compliance'] == q]
        axes[0].plot([r['damage_cap'] for r in subset], [r['max_difference_m']*1000 for r in subset], 'o-', label=f'q={q}')
    axes[0].axhline(.020, color='gray', ls='--', label='0.020 mm (not an impossibility threshold)')
    axes[0].set(xlabel='Cumulative added damage cap', ylabel='Pair-only maximum gap difference (mm)')
    axes[0].legend(fontsize=7)
    for name in ('repeat', 'zero_probe_alternating'):
        for tau, style in ((0., '-'), (settings['gaussian_bias_sigma_m'], '--')):
            subset = [r for r in sequences if r['matched_compliance'] == '1/2' and r['sigma_m'] == .00002 and r['schedule'] == name and r['bias_sigma_m'] == tau]
            axes[1].plot([r['count'] for r in subset], [r['equal_prior_bayes_error'] for r in subset], 'o'+style,
                label=f'{name}, bias SD={tau*1000:g} mm')
    axes[1].set(xscale='log', xlabel='Total observations', ylabel='Oracle Gaussian binary Bayes error', ylim=(-.01, .51))
    axes[1].legend(fontsize=7)
    fig.suptitle('Designed two-state diagnostic: not a full-belief achievable policy')
    fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=170); plt.close(fig)
    print(json.dumps({'output': str(output), 'pair_rows': len(rows), 'gaussian_rows': len(sequences),
        'filtered_prefixes': [{k:r[k] for k in ('matched_compliance','sigma_m','selected_action_m','pair_difference_at_selected_m','proof_replayed')} for r in filtered_rows],
        'cap_002': [{k:r[k] for k in ('matched_compliance','pair_safe_ceiling_m','max_difference_m')} for r in rows if r['damage_cap'] == .02]}, indent=2))


if __name__ == '__main__':
    main()
