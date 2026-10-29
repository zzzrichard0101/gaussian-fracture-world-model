from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.continuous_risk import replay_proof
from wmfracture.observation_budget import bounded_noise, BudgetProposal
from wmfracture.verified_policy import filter_proposal, RiskReuseSession


def configuration():
    root=Path(__file__).parents[1]
    settings=json.loads((root/'config/observation-budget.json').read_text())
    guard=json.loads((root/'config/finite-damage-guard.json').read_text());guard['grid_side']=7
    return settings,guard


def test_noise_is_reproducible_bounded_and_has_matching_marginals_with_correlation():
    iid=bounded_noise(73,30000,.00002,3.,0.)
    correlated=bounded_noise(73,30000,.00002,3.,.8)
    assert np.array_equal(iid,bounded_noise(73,30000,.00002,3.,0.))
    for noise in (iid,correlated):
        assert np.max(np.abs(noise))<=.00006
        assert abs(np.mean(noise))<.000001
        assert .000018<np.std(noise)<.000021
    assert abs(np.corrcoef(iid[:-1],iid[1:])[0,1])<.04
    assert np.corrcoef(correlated[:-1],correlated[1:])[0,1]>.7


@pytest.mark.parametrize('rho',[-1.,1.,float('nan')])
def test_invalid_noise_correlation_rejected(rho):
    with pytest.raises(ValueError):bounded_noise(1,10,.001,3.,rho)


@pytest.mark.parametrize('policy',['rest_repeat','weak_repeat','fixed_ramp','adaptive_verified'])
def test_proposals_need_only_public_prefix_and_keep_cumulative_reference(policy):
    settings,guard=configuration();law,fixture=InterfaceLaw(area=.01),Fixture()
    proposer=BudgetProposal(policy,settings,guard,law,fixture,.00002)
    prefix={'actions_m':[0.,.001],'observed_gaps_m':[0.,.0005]}
    action,record=proposer.plan(prefix,0)
    assert action>=0. and isinstance(record,dict)
    data={'parameter_bounds':guard['parameter_bounds'],'spring':fixture.spring,
        'law':{'area':law.area,'tensile_stiffness':law.tensile_stiffness,'peak_traction':law.peak_traction},
        **deepcopy(prefix),'reference_count':2,'noise_radius_m':.00006,'bias_bound_m':.00015,'damage_cap':.02}
    session=RiskReuseSession()
    result=filter_proposal(data,action,lambda a:[(2.,.0016)],max_cells=127,backtracking_steps=2,reuse_session=session)
    assert not result['stopped'] and replay_proof(result['proof'])
    assert session.search.inputs['reference_count']==2
    data['actions_m'].append(result['action_m']);data['observed_gaps_m'].append(.0005)
    session.context(data)
    assert session.search.reference_peak==.001


def test_fixed_plan_does_not_depend_on_future_budget_or_observation_values():
    settings,guard=configuration();proposer=BudgetProposal('fixed_ramp',settings,guard,InterfaceLaw(),Fixture(),.00002)
    assert len(settings['fixed_proposals_m'])==max(settings['budgets'])
    for turn in range(max(settings['budgets'])):
        a={'actions_m':[0.,.001]+[0.]*turn,'observed_gaps_m':[0.]*(turn+2)}
        b=deepcopy(a);b['observed_gaps_m']=[1.]*(turn+2)
        assert proposer.plan(a,turn)[0]==proposer.plan(b,turn)[0]==settings['fixed_proposals_m'][turn]


def experiment_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import observation_budget_evaluate
    return observation_budget_evaluate


def test_nested_checkpoints_sealed_public_forecasts_and_cold_proof_replay(tmp_path,monkeypatch):
    module=experiment_module(monkeypatch);settings,guard=configuration()
    settings.update(budgets=[2,4],policies=['rest_repeat','weak_repeat'],gap_noise_sigma_m=[.00002],
        actual_bias_m=[.00003],noise_conditions=[{'name':'iid_bounded','latent_rho':0.}],
        grid_side=7,forecast_grid_sides=[7,9],primary_forecast_side=9,cell_budget=127,backtracking_steps=2)
    cases=[{'id':'test','parameters':[2.,.0016],'family':'test','cluster':'test'}]
    (tmp_path/'proofs').mkdir();law,fixture=InterfaceLaw(area=.01),Fixture()
    for name,value in [('config.json',settings),('guard-config.json',guard),('private-cases.json',cases),('code-sha256.json',{})]:
        module.write_json(tmp_path/name,value)
    module.collect(tmp_path,settings,guard,cases,law,fixture)
    public=json.loads((tmp_path/'public-episodes.json').read_text())
    for policy in settings['policies']:
        short=next(r for r in public if r['policy']==policy and r['budget']==2)
        long=next(r for r in public if r['policy']==policy and r['budget']==4)
        assert short['actions_m']==long['actions_m'][:4]
        assert short['observed_gaps_m']==long['observed_gaps_m'][:4]
    original_read=Path.read_text
    def protected_read(path,*args,**kwargs):
        if path.name.startswith('private-'):raise AssertionError('Forecast read private labels')
        return original_read(path,*args,**kwargs)
    with monkeypatch.context() as protected:
        protected.setattr(Path,'read_text',protected_read)
        seals=module.forecast_stage(tmp_path,settings,guard,law,fixture)
    groups,rows,replayed=module.evaluate(tmp_path,settings,guard,law,fixture,seals)
    assert len(rows)==8 and replayed==8
    assert not any(r['cap_violation'] or not r['budget_fulfilled'] for r in rows)


def test_stopped_policy_keeps_actual_observations_and_does_not_fill_budget(tmp_path,monkeypatch):
    module=experiment_module(monkeypatch);settings,guard=configuration()
    settings.update(budgets=[2,4],policies=['rest_repeat'],gap_noise_sigma_m=[.00002],
        actual_bias_m=[0.],noise_conditions=[{'name':'iid_bounded','latent_rho':0.}])
    def stopped_decision(root,identifier,turn,*args):
        return {'action_m':0. if turn<2 else None,'stopped':turn>=2,'policy_seconds':0.}
    monkeypatch.setattr(module,'decide',stopped_decision)
    cases=[{'id':'test','parameters':[2.,.0016],'family':'test','cluster':'test'}]
    module.collect(tmp_path,settings,guard,cases,InterfaceLaw(area=.01),Fixture())
    short,long=json.loads((tmp_path/'public-episodes.json').read_text())
    assert short['budget_fulfilled'] and not short['stopped']
    assert not long['budget_fulfilled'] and long['stopped']
    assert long['actions_m']==short['actions_m'] and long['delivered_observations']==2


def test_paired_cluster_summary_keeps_matched_pairs_together(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from report_observation_budget import paired_contrast
    left={str(i):{'cluster':str(i//2),'future_mean_squared_error_mm2':1.} for i in range(8)}
    right={str(i):{'cluster':str(i//2),'future_mean_squared_error_mm2':4.} for i in range(8)}
    plan={'bootstrap_seed':1,'bootstrap_repetitions':100}
    result=paired_contrast(left,right,plan)
    assert result['designed_clusters']==4 and result['paired_contexts']==8
    assert result['delta_rmse_mm']==-1.
    assert result['cluster_bootstrap_95_percentile_delta_mm']==[-1.,-1.]
    assert paired_contrast(left,left,plan)['cluster_bootstrap_95_percentile_delta_mm']==[0.,0.]
