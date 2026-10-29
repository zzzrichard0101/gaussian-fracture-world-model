from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
import pytest

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import normalize
from wmfracture.measurement_bias import GaussianObservation
from wmfracture.guided_quadrature import GuidedResponseQuadrature
from wmfracture.continuous_risk import replay_proof
from wmfracture.task_design import expected_task_gain,compress_mixture,PredictiveMixture,TaskController,risk_inputs


def settings():
    root=Path(__file__).parents[1]
    s=json.loads((root/'config/task-aware-pilot.json').read_text())
    s.update(cell_budget=127,backtracking_steps=2)
    g=json.loads((root/'config/finite-damage-guard.json').read_text());g['grid_side']=9
    return s,g


def test_gain_matches_linear_gaussian_bayes_variance_reduction():
    nodes,weights=hermgauss(64);theta=nodes*np.sqrt(2);weights=weights/np.sqrt(np.pi)
    result=expected_task_gain(theta,weights,theta[:,None],1.,bin_std=.0625,outcome_order=12)
    assert result['valid']
    assert result['gain_mm2']==pytest.approx(.5,abs=.002)
    assert result['expected_posterior_variance_mm2']==pytest.approx(.5,abs=.002)


def test_identical_observations_and_constant_target_have_zero_gain():
    weights=np.array([.25,.75])
    for means,targets in [(np.zeros(2),np.array([[-1.],[1.]])),(np.array([-2.,2.]),np.ones((2,1)))]:
        result=expected_task_gain(means,weights,targets,1.)
        assert result['valid'] and result['gain_mm2']<1e-25


def test_mixture_compression_preserves_mass_measurement_and_target_means():
    rng=np.random.default_rng(5);means=rng.normal(size=1000);weights=rng.random(1000);weights/=weights.sum()
    targets=rng.normal(size=(1000,3))
    mu,mass,f=compress_mixture(means,weights,targets,1.,.25,4096)
    assert mass.sum()==pytest.approx(1.)
    assert mass@mu==pytest.approx(weights@means,abs=1e-14)
    np.testing.assert_allclose(mass@f,weights@targets,atol=1e-14)


def test_shared_bias_predictive_update_matches_full_prefix_likelihood():
    s,g=settings();law,fixture=InterfaceLaw(area=.01),Fixture()
    prefix={'actions_m':[0.,.001],'observed_gaps_m':[.00003,.00063]};sigma=.00002
    model=GuidedResponseQuadrature(law,fixture,g['parameter_bounds'],prefix['actions_m'],prefix['observed_gaps_m'],sigma,g['bias_sigma_m'])
    belief=model.integrate(s['future_drives_m'],energy_panels=4,inner_order=8)
    mixture=PredictiveMixture(model,belief,s);action=.0013;observed=.00081
    conditional=-.5*(observed-mixture.means(action))**2/mixture.noise_variance
    old_prediction=model.predict(mixture.parameters,prefix['actions_m'])
    old=GaussianObservation('shared',g['bias_sigma_m'])(old_prediction,prefix['observed_gaps_m'],sigma)
    full=model.predict(mixture.parameters,prefix['actions_m']+[action])
    updated=GaussianObservation('shared',g['bias_sigma_m'])(full,prefix['observed_gaps_m']+[observed],sigma)
    np.testing.assert_allclose(normalize(old+conditional),normalize(updated),atol=1e-13,rtol=1e-10)


@pytest.mark.parametrize('policy',['observation_disagreement','future_task'])
def test_controller_scores_only_certified_actual_inputs_and_proof_binds_context(policy):
    s,g=settings();law,fixture=InterfaceLaw(area=.01),Fixture()
    prefix={'actions_m':[0.,.001],'observed_gaps_m':[0.,.0005]};original=deepcopy(prefix)
    controller=TaskController(policy,law,fixture,g,s,.00002)
    action,proof,record=controller.plan(prefix)
    assert prefix==original and action is not None
    assert replay_proof(proof,expected_inputs=risk_inputs(prefix,.00002,g,law,fixture),expected_action=action)
    assert action in [r['action_m'] for r in record['scores']] or record['acquisition_abstained']
    assert proof['inputs']['reference_count']==2
    assert all(r['score'] is not None for r in record['scores']) or record['acquisition_abstained']


def test_acquisition_budget_failure_is_not_an_unsafe_action_certificate():
    s,g=settings();s['acquisition_max_bins']=1
    model=GuidedResponseQuadrature(InterfaceLaw(area=.01),Fixture(),g['parameter_bounds'],[0.,.001],[0.,.0005],.00002,g['bias_sigma_m'])
    result=model.integrate(s['future_drives_m'],energy_panels=4,inner_order=8)
    mixture=PredictiveMixture(model,result,s)
    score,diagnostic=mixture.task_score(.0013,s)
    assert score is None and not diagnostic['stable']


def test_task_gain_prefers_target_information_over_larger_nuisance_variance():
    target=np.array([-1.,-1.,1.,1.]);nuisance=np.array([-2.,2.,-2.,2.]);weights=np.full(4,.25)
    task=expected_task_gain(target,weights,target[:,None],1.)
    irrelevant=expected_task_gain(nuisance,weights,target[:,None],1.)
    assert np.var(nuisance)>np.var(target)
    assert task['gain_mm2']>.3 and irrelevant['gain_mm2']<1e-20


def test_small_closed_loop_keeps_fixed_damage_reference_and_seals_predictions(tmp_path,monkeypatch):
    root=Path(__file__).parents[1];monkeypatch.syspath_prepend(str(root/'scripts'))
    import task_aware_pilot as pilot
    from response_quadrature_evaluate import forecast_public
    from history_refinement import write_json
    s,g=settings();base=json.loads((root/'config/observation-budget.json').read_text());base.update(s);s=base
    s.update(policies=['calibration_alternating','future_task'],budgets=[2],gap_noise_sigma_m=[.00002],actual_bias_m=[.00003],
        noise_conditions=[{'name':'iid_bounded','latent_rho':0.}],levels=[{'name':'guided_32',**s['belief_levels'][1]},
            {'name':'guided_64',**s['belief_levels'][2]}],primary_level='guided_64',event_mass_change_tolerance=s['event_change_tolerance'],
        adaptive_audit_case_indices=[],adaptive_audit_budget=8,adaptive_audit_actual_bias_m=.00003)
    case_settings=json.loads((root/'config/continuous-risk-evaluation.json').read_text())
    case_settings.update(seed=s['case_seed'],families=s['case_families'])
    law,fixture=InterfaceLaw(area=.01),Fixture();cases=pilot.new_cases(case_settings,g,law,fixture,[])[:1]
    (tmp_path/'proofs').mkdir()
    for name,value in [('config.json',s),('code-sha256.json',{}),('private-cases.json',cases)]:write_json(tmp_path/name,value)
    pilot.collect(tmp_path,s,g,cases,law,fixture)
    public=json.loads((tmp_path/'public-episodes.json').read_text())
    assert len(public)==2 and all(e['budget_fulfilled'] for e in public)
    bundle={'source_run_id':'test','episodes':public,'baseline_forecasts':[],
        'source_settings':s,'guard':g,'case_settings':case_settings,'law':pilot.asdict_law(law)}
    write_json(tmp_path/'inputs.json',bundle)
    seals=forecast_public(tmp_path,bundle,s,engine_class=GuidedResponseQuadrature)
    groups,rows,replayed=pilot.evaluate_risk(tmp_path,bundle,s,g,law,fixture,seals)
    assert replayed==4 and len(rows)==4 and not any(r['cap_violation'] for r in rows)
    for name in (tmp_path/'proofs').iterdir():
        import gzip
        with gzip.open(name,'rt') as stream:assert json.load(stream)['inputs']['reference_count']==2


def test_adaptive_controller_does_not_read_label_generation_metadata():
    s,g=settings()
    class PublicSettings(dict):
        def __getitem__(self,key):
            if key in ('seed','case_seed','case_families','primary_comparison'):
                raise AssertionError('Controller accessed label/analysis metadata')
            return super().__getitem__(key)
    controller=TaskController('future_task',InterfaceLaw(area=.01),Fixture(),g,PublicSettings(s),.00002)
    action,proof,_=controller.plan({'actions_m':[0.,.001],'observed_gaps_m':[0.,.0005]})
    assert action is not None and replay_proof(proof)
