"""Paired public-trace replay; identical proofs, no new private-state evaluation."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from history_refinement import digest, write_json
from continuous_risk_probe import public_inputs, WarmBank
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.damage_guard import FiniteGuard
from wmfracture.continuous_risk import replay_proof
from wmfracture.verified_policy import filter_proposal, RiskReuseSession


def semantic(proof):
    return {k:v for k,v in proof.items() if k not in ('point_evaluations','reuse_stats')}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def attempt_signature(attempts):
    fields=('action_m','decision','lower_damage','upper_damage','cells','witness')
    return [{k:a[k] for k in fields} for a in attempts]


def prepare_inputs(source):
    # Only public observations, known settings and old verifier outputs are read.
    # No private parameters, traces, labels or forecasts are used.
    settings=json.loads((source/'config.json').read_text())
    config=json.loads((source/'guard-config.json').read_text())
    law=json.loads((source/'law-config.json').read_text())['law']
    steps=json.loads((source/'decisions.json').read_text());episodes={}
    for step in steps:
        if step['policy']!='verified_filter':continue
        proof=json.loads((source/step['executed_proof']['file']).read_text())
        assert digest(source/step['executed_proof']['file'])==step['executed_proof']['sha256']
        episodes.setdefault(step['episode_id'],[]).append({
            **{k:step[k] for k in ('round','public_prefix','sigma_m','proposal_m','action_m')},
            'attempt_signature':attempt_signature(step['filter']['attempts']),
            'proof_semantic_sha256':fingerprint(semantic(proof))})
    names=('config.json','guard-config.json','law-config.json','decisions.json')
    return {'settings':settings,'guard_config':config,'law':law,
        'source_run_id':source.name,'source_sha256':{n:digest(source/n) for n in names},
        'episodes':episodes,'scope':'Public trace replay only; no new predictive or safety evidence.'}


def run_episode(identifier,steps,mode,bundle,root,repeat):
    settings=bundle['settings'];config=bundle['guard_config']
    law=InterfaceLaw(**bundle['law']);fixture=Fixture(**settings['fixture'])
    controller=FiniteGuard(law,fixture,config,steps[0]['sigma_m'])
    session=RiskReuseSession() if mode=='reuse' else None
    rows=[]
    for step in steps:
        start=time.perf_counter()
        proposal,_=controller.plan(step['public_prefix']['actions_m'],step['public_prefix']['observed_gaps_m'],
            'continuous_guard',step['round'])
        assert proposal==step['proposal_m']
        data=public_inputs(step['public_prefix'],step['sigma_m'],config,law,fixture,2)
        bank=WarmBank(data,settings['grid_side'],law,fixture)
        result=filter_proposal(data,proposal,lambda a:bank.points(a,settings['warm_point_count']),
            max_cells=settings['cell_budget'],backtracking_steps=settings['backtracking_steps'],reuse_session=session)
        path=root/'proofs'/f'{repeat}-{mode}-{identifier}-{step["round"]}.json'
        write_json(path,result['proof'])
        elapsed=time.perf_counter()-start
        # Equality assertions and extra cold replay are outside the timed policy.
        assert result['action_m']==step['action_m']
        assert attempt_signature(result['attempts'])==step['attempt_signature']
        assert fingerprint(semantic(result['proof']))==step['proof_semantic_sha256']
        attempts=result['attempts']
        constraints={}
        for name,field in (('search','reuse_stats'),('checker','replay_stats')):
            for key in ('box_constraints','point_constraints','box_hits','point_hits','box_extensions','point_extensions'):
                constraints[f'{name}_{key}']=sum(a[field][key] for a in attempts)
        rows.append({'episode_id':identifier,'round':step['round'],'repeat':repeat,'mode':mode,
            'policy_seconds':elapsed,'solve_seconds':sum(a['solve_seconds'] for a in attempts),
            'replay_seconds':sum(a['proof_replay_seconds'] for a in attempts),
            'logical_cells':sum(a['cells'] for a in attempts),'queries':len(attempts),
            'proof_file':str(path.relative_to(root)),'proof_sha256':digest(path),**constraints})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path)
    parser.add_argument('--repetitions',type=int,default=2)
    args=parser.parse_args();assert args.repetitions>0
    project=Path(os.environ['WM_PROJECT_ROOT'])
    source=None
    if args.inputs:
        bundle=json.loads(args.inputs.read_text())
    else:
        report=json.loads((project/'artifacts/continuous-risk-evaluation/report.json').read_text())
        source=Path(report['run_directory']);bundle=prepare_inputs(source)
    root=Path(os.environ['WM_RESEARCH_ROOT'])/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-risk-reuse')
    (root/'proofs').mkdir(parents=True);print(f'OUTPUT={root}',flush=True)
    write_json(root/'inputs.json',bundle)
    write_json(root/'config.json',{'repetitions':args.repetitions,'mode_order':'alternating per episode, reversed each repetition',
        'timing':'proposal, public warm bank, all solves/replays, selected proof JSON write; excludes equality checks and final cold replay',
        'scope':'Same existing public traces; exact semantic equivalence, engineering cache baseline; no novelty claim.'})
    paths=[Path(__file__),project/'wmfracture/continuous_risk.py',project/'wmfracture/verified_policy.py',
        project/'scripts/continuous_risk_probe.py',project/'wmfracture/damage_guard.py',
        project/'wmfracture/action_ceiling.py',project/'wmfracture/compliant_inference.py',
        project/'wmfracture/cohesive.py',project/'wmfracture/compliant_fixture.py',
        project/'wmfracture/measurement_bias.py',project/'wmfracture/joint_state.py',project/'scripts/history_refinement.py']
    for p in paths:
        target=root/'source'/p.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write_json(root/'code-sha256.json',{str(p.relative_to(project)):digest(p) for p in paths})
    rows=[];start=time.perf_counter()
    for repeat in range(args.repetitions):
        for i,(identifier,steps) in enumerate(bundle['episodes'].items()):
            modes=('cold','reuse') if (i+repeat)%2==0 else ('reuse','cold')
            for mode in modes:rows+=run_episode(identifier,steps,mode,bundle,root,repeat)
            if (i+1)%24==0:print(f'REPLAY repeat={repeat+1} episodes={i+1}/{len(bundle["episodes"])}',flush=True)
    measured_seconds=time.perf_counter()-start
    # Fresh checker, no memoization state from either search or paired replay.
    checked=0;start=time.perf_counter()
    for row in rows:
        if row['mode']!='reuse':continue
        path=root/row['proof_file'];assert digest(path)==row['proof_sha256']
        proof=json.loads(path.read_text());step=bundle['episodes'][row['episode_id']][row['round']]
        law=InterfaceLaw(**bundle['law']);fixture=Fixture(**bundle['settings']['fixture'])
        data=public_inputs(step['public_prefix'],step['sigma_m'],bundle['guard_config'],law,fixture,2)
        assert replay_proof(proof,expected_inputs=data,expected_action=step['action_m']);checked+=1
    cold_replay_seconds=time.perf_counter()-start
    groups=[]
    for mode in ('cold','reuse'):
        chosen=[r for r in rows if r['mode']==mode]
        episode_seconds=[sum(r['policy_seconds'] for r in chosen if r['episode_id']==e and r['repeat']==rep)
            for rep in range(args.repetitions) for e in bundle['episodes']]
        groups.append({'mode':mode,'episode_replays':len(episode_seconds),
            'mean_two_step_ms':float(np.mean(episode_seconds)*1000),'median_two_step_ms':float(np.median(episode_seconds)*1000),
            'p95_two_step_ms':float(np.quantile(episode_seconds,.95)*1000),
            **{key:sum(r[key] for r in chosen) for key in ('policy_seconds','solve_seconds','replay_seconds','logical_cells','queries',
                'search_box_constraints','search_point_constraints','checker_box_constraints','checker_point_constraints',
                'search_box_hits','checker_box_hits','search_box_extensions','checker_box_extensions')}})
    assert groups[0]['logical_cells']==groups[1]['logical_cells'] and groups[0]['queries']==groups[1]['queries']
    source_unchanged=None if source is None else all(digest(source/n)==h for n,h in bundle['source_sha256'].items())
    assert source_unchanged is not False
    report={'run_directory':str(root),'source_run_id':bundle['source_run_id'],'groups':groups,
        'unique_source_episodes':len(bundle['episodes']),'repetitions':args.repetitions,'cold_replayed_reuse_proofs':checked,
        'all_actions_attempt_decisions_bounds_witnesses_and_selected_proof_trees_equal':True,
        'speedup_ratio':groups[0]['policy_seconds']/groups[1]['policy_seconds'],
        'timed_replays_and_assertions_wall_seconds':measured_seconds,'extra_cold_replay_seconds':cold_replay_seconds,
        'original_source_unchanged':source_unchanged,'predictive_results_reused_not_remeasured':True,
        'new_method_novelty_validated':False,'inputs_sha256':digest(root/'inputs.json')}
    write_json(root/'rows.json',rows);write_json(root/'report.json',report)
    out=project/'artifacts/risk-reuse';out.mkdir(parents=True,exist_ok=True)
    for name in ('inputs.json','rows.json','report.json','config.json','code-sha256.json'):shutil.copy2(root/name,out/name)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
