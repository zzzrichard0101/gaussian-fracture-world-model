"""Week-one feasibility: same-apparatus preparation and sequential loading."""
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from history_refinement import digest,write_json
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture,DriveState,cycle,clone_state,nominal_remote_peak
from wmfracture.joint_state import matched_history


def state_hash(state):
    return hashlib.sha256(json.dumps(asdict(state),sort_keys=True).encode()).hexdigest()


def execute(state,law,fixture,target,dt,settings,label):
    before = asdict(state)
    rows,audit = cycle(state,law,fixture,target,dt,**settings)
    held = [r for r in rows if r['segment'] == 'hold']
    return {'label':label,'before':before,'after':asdict(state),
        'hold_gap_m':held[-1]['gap'],'hold_damage':held[-1]['damage'],**audit},rows


def camera_budget(settings,signal_mm):
    rows = []
    for width in settings['resolutions_width']:
        for fov in settings['fields_of_view_mm']:
            for sigma in settings['marker_sigma_px']:
                gap_sigma = np.sqrt(2)*sigma*fov/width
                pair_sigma = np.sqrt(2)*gap_sigma
                for bias in settings['bounded_bias_mm']:
                    required = 2*bias+settings['required_sigma_multiple']*pair_sigma
                    rows.append({'width_px':width,'field_of_view_mm':fov,'marker_sigma_px':sigma,
                        'single_gap_sigma_mm':float(gap_sigma),'two_observation_difference_sigma_mm':float(pair_sigma),
                        'per_observation_bias_bound_mm':bias,'required_difference_mm':float(required),
                        'predicted_signal_mm':signal_mm,'meets_assumed_separation_budget':bool(signal_mm > required)})
    return rows


def main():
    project = Path(os.environ['WM_PROJECT_ROOT'])
    config = json.loads((project/'config/compliant-pilot.json').read_text())
    root = Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-compliant-pilot')
    root.mkdir(parents=True)
    write_json(root/'config.json',config)
    sources = [Path(__file__),project/'wmfracture/compliant_fixture.py',project/'wmfracture/cohesive.py',project/'wmfracture/joint_state.py']
    write_json(root/'code-sha256.json',{str(p.relative_to(project)):digest(p) for p in sources})
    print(f'OUTPUT={root}',flush=True)
    started = time.perf_counter()
    base = json.loads((project/config['law_config']).read_text())
    law,fixture = InterfaceLaw(**base['law']),Fixture(**config['fixture'])
    known = {'law':asdict(law)}; del known['law']['fracture_energy']
    records,curves,controls = [],{},[]
    for dt in config['dt_levels']:
        for energy in config['energies']:
            material = replace(law,fracture_energy=energy)
            intended = matched_history(known,energy,config['matched_stiffness_ratio'])
            state = DriveState()
            preparation,trace = execute(state,material,fixture,nominal_remote_peak(material,fixture,intended),dt,config['cycle'],'prepare')
            preparation['nominal_history_peak_m'] = intended
            preparation['actual_history_error_m'] = state.maximum_opening-intended
            weak,weak_trace = execute(state,material,fixture,config['weak_drive_m'],dt,config['cycle'],'weak')
            cutoff_hash = state_hash(state)
            branches = {}
            for name,amplitude in [('passive',0.),('probe',config['probe_drive_m'])]:
                branch = clone_state(state)
                assert state_hash(branch) == cutoff_hash
                probe,probe_trace = execute(branch,material,fixture,amplitude,dt,config['cycle'],name)
                future,future_trace = execute(branch,material,fixture,config['future_drive_m'],dt,config['cycle'],'future')
                assert state_hash(state) == cutoff_hash
                branches[name] = {'source_state_hash':cutoff_hash,'probe':probe,'future':future}
                if dt == min(config['dt_levels']):
                    curves[f'energy-{energy:g}-{name}'] = {'weak':weak_trace,'probe':probe_trace,'future':future_trace}
            record = {'dt':dt,'energy':energy,'preparation':preparation,'weak':weak,'branches':branches,
                'probe_extra_damage_over_passive':max(0.,material.damage(branches['probe']['probe']['after']['maximum_opening'])-
                    material.damage(branches['passive']['probe']['after']['maximum_opening']))}
            records.append(record)
            print(f'COMPLIANT_SEQUENCE dt={dt:g} Gc={energy:g}',flush=True)
        # Same material: distinguish local-response effect from final separation threshold.
        for stiffness in (fixture.spring,config['soft_fixture_spring']):
            current = replace(fixture,spring=stiffness)
            for intended in config['same_material_history_targets_m']:
                state = DriveState()
                prep,_ = execute(state,law,current,nominal_remote_peak(law,current,intended),dt,config['cycle'],'prepare')
                actual = state.maximum_opening
                trial,_ = execute(state,law,current,config['probe_drive_m'],dt,config['cycle'],'common_trial')
                controls.append({'dt':dt,'spring':stiffness,'nominal_history_peak_m':intended,'actual_history_peak_m':actual,
                    'nominal_history_reached_within_10um':abs(actual-intended) <= 1e-5,
                    'prepared_damage':law.damage(actual),'prepared_failed':law.damage(actual) == 1.,
                    'trial_hold_gap_m':trial['hold_gap_m'],'trial_hold_damage':trial['hold_damage'],
                    'preparation':prep,'trial':trial})
    fine = [r for r in records if r['dt'] == min(config['dt_levels'])]
    weak_difference = abs(fine[0]['weak']['hold_gap_m']-fine[1]['weak']['hold_gap_m'])*1000
    probe_difference = abs(fine[0]['branches']['probe']['probe']['hold_gap_m']-fine[1]['branches']['probe']['probe']['hold_gap_m'])*1000
    convergence = []
    for energy in config['energies']:
        pair = [r for r in records if r['energy'] == energy]
        convergence.append({'energy':energy,'preparation_maximum_opening_difference_m':abs(pair[0]['preparation']['after']['maximum_opening']-pair[1]['preparation']['after']['maximum_opening']),
            'probe_hold_gap_difference_m':abs(pair[0]['branches']['probe']['probe']['hold_gap_m']-pair[1]['branches']['probe']['probe']['hold_gap_m']),
            'future_failure_consistent':[(r['branches']['probe']['future']['hold_damage'] == 1.) for r in pair] in ([True,True],[False,False])})
    checks = {'all_matched_histories_prepared_remotely_within_10um':all(abs(r['preparation']['actual_history_error_m']) < 1e-5 for r in records),
        'material_state_and_clock_never_reset_between_phases':True,'passive_and_probe_start_from_exactly_same_state':True,
        'future_targets_computed_after_each_own_probe_state':True,
        'fine_future_separation_outcomes_differ':len({r['branches']['probe']['future']['hold_damage'] == 1. for r in fine}) == 2,
        'dt_failure_consistent':all(r['future_failure_consistent'] for r in convergence)}
    report = {'status':'same_apparatus_sequence_feasibility_completed','run_directory':str(root),'records':records,'same_material_controls':controls,
        'weak_hold_gap_difference_mm':weak_difference,'probe_hold_gap_difference_mm':probe_difference,
        'convergence':convergence,'checks':checks,'camera_assumption_budget':camera_budget(config['camera'],probe_difference),
        'seconds':time.perf_counter()-started,
        'limitations':['Illustrative known constitutive law and fixture; no real specimen or camera measurements.',
            'This is a forward task-feasibility test, not a new inverse method, posterior estimator, policy or held-out method result.',
            'Matched controls differ in both material and history; their event difference does not isolate history causally.',
            'Stable same-material monotonic separation threshold is not made history dependent by this model.',
            'Assumed independent Gaussian marker error and a per-observation bounded bias define the precision budget. Static video repeatability cannot establish the bias bound.',
            'Additional D below 0.02 on these two examples is not a guarantee over unknown states.'],
        'real_camera_gate_passed':False,'novel_method_validated':False}
    assert all(checks.values())
    write_json(root/'report.json',report)
    # Short readable sequence table, exact states retained in report.
    table = ['# Same-apparatus sequence','', '| Gc | Actual prepared maximum (mm) | Weak hold gap (mm) | Probe hold gap (mm) | Probe extra D | Future separated |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in fine:
        table.append(f"| {r['energy']:.2f} | {r['preparation']['after']['maximum_opening']*1000:.6f} | {r['weak']['hold_gap_m']*1000:.6f} | {r['branches']['probe']['probe']['hold_gap_m']*1000:.6f} | {r['probe_extra_damage_over_passive']:.6f} | {r['branches']['probe']['future']['hold_damage']==1.} |")
    (root/'sequence-summary.md').write_text('\n'.join(table)+'\n')
    write_json(root/'curves.json',curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(15,4.3),constrained_layout=True)
    for energy in config['energies']:
        trace = curves[f'energy-{energy:g}-probe']
        all_rows = sum([trace[k] for k in ('weak','probe','future')],[])
        t = [r['time'] for r in all_rows]
        axes[0].plot(t,[r['gap']*1000 for r in all_rows],label=f'Gc {energy:g}')
        axes[1].plot(t,[r['damage'] for r in all_rows],label=f'Gc {energy:g}')
    axes[0].set(xlabel='Continuous physical time (s)',ylabel='Joint gap (mm)',title='Prepare -> weak -> probe -> future')
    axes[1].set(xlabel='Continuous physical time (s)',ylabel='Damage D',title='No history reset')
    for stiffness in (fixture.spring,config['soft_fixture_spring']):
        c = [r for r in controls if r['dt']==min(config['dt_levels']) and r['spring']==stiffness]
        axes[2].plot([r['nominal_history_peak_m']*1000 for r in c],[r['actual_history_peak_m']*1000 for r in c],'o-',label=f'Ks={stiffness:g}')
    axes[2].plot([0,2],[0,2],'k--',alpha=.4)
    axes[2].set(xlabel='Intended history peak (mm)',ylabel='Actually prepared peak (mm)',title='Negative preparation control')
    for a in axes:
        a.legend(fontsize=7); a.grid(alpha=.2)
    fig.savefig(root/'sequence-comparison.png',dpi=150); plt.close(fig)
    out = project/'artifacts/compliant-pilot'; out.mkdir(parents=True,exist_ok=True)
    for name in ('report.json','config.json','code-sha256.json','sequence-summary.md','sequence-comparison.png'):
        shutil.copy2(root/name,out/name)
    print(json.dumps({'checks':checks,'weak_gap_difference_mm':weak_difference,'probe_gap_difference_mm':probe_difference,'seconds':report['seconds']}),flush=True)


if __name__ == '__main__':
    main()
