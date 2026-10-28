import numpy as np
import pytest

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import equilibrium_cycle, damage
from wmfracture.damage_guard import (feasible_bias_interval,action_risks,admissibility,
    risk_witnesses,FiniteGuard)


def settings():
    return {'parameter_bounds':[[1.2,3.],[.0009,.0023]],'grid_side':11,'subset_size':8,
        'damage_cap':.02,'noise_multiple':3.,'bias_bound_m':.00015,'bias_sigma_m':.00005,
        'weak_drive_m':.001,'actions_m':[0.,.001,.0012,.00135,.0015,.00165,.00185],
        'fixed_probes_m':[.00165,.00185],'minimum_information_score':1e-8}


def test_persistent_bias_must_fit_all_observations_with_one_value():
    prediction = np.array([[0.,.001],[0.,.001]])
    observed = np.array([.00003,.00103])
    mask,lower,upper = feasible_bias_interval(prediction,observed,.000005,3.,.00015)
    assert mask.all() and np.all(lower <= .00003) and np.all(upper >= .00003)
    mask,_,_ = feasible_bias_interval(prediction,[.00006,.00094],.000005,3.,.00015)
    assert not mask.any()  # two individually plausible but incompatible offsets


def test_added_observation_only_shrinks_feasible_parameter_set():
    prediction = np.array([[0.,.001,.0015],[0.,.0005,.0009],[0.,.001,.001]])
    first,_,_ = feasible_bias_interval(prediction[:,:2],[.00003,.00103],.00001,3.,.00015)
    second,_,_ = feasible_bias_interval(prediction,[.00003,.00103,.00153],.00001,3.,.00015)
    assert not np.any(second & ~first) and first.sum() > second.sum()


def test_damage_cap_is_cumulative_not_a_reset_per_action():
    law,fixture = InterfaceLaw(area=.01),Fixture()
    energy = np.array([2.]); baseline = np.array([.001])
    _,one = equilibrium_cycle(baseline,energy,.00194,law,fixture)
    first,_ = action_risks(baseline,baseline,energy,[.00194],law,fixture)
    _,two = equilibrium_cycle(one,energy,.00210,law,fixture)
    second,_ = action_risks(one,baseline,energy,[.00210],law,fixture)
    local_increment = damage(two,energy,law)-damage(one,energy,law)
    assert first[0,0] < .02 and local_increment[0] < .02 and second[0,0] > .02


def test_top_weight_sample_can_miss_risk_and_witness_restores_decisions():
    risks = np.array([[0.,0.,0.],[.001,.002,.08],[.001,.003,.01]])
    failures = np.zeros_like(risks,dtype=bool)
    full = admissibility(risks,failures,.02)
    subset = admissibility(risks[:,:2],failures[:,:2],.02)
    assert subset[1] and not full[1]
    witnesses = risk_witnesses(risks,failures,.02)
    assert witnesses.tolist() == [2]
    union = np.unique(np.r_[0,1,witnesses])
    np.testing.assert_array_equal(admissibility(risks[:,union],failures[:,union],.02),full)
    assert not admissibility(np.empty((3,0)),np.empty((3,0),bool),.02).any()


def test_new_separation_is_forbidden_even_when_additional_damage_is_small():
    risks = np.array([[.005,.01]])
    failures = np.array([[True,False]])
    assert not admissibility(risks,failures,.02)[0]
    witness = risk_witnesses(risks,failures,.02)
    assert witness.tolist() == [0]


def test_full_and_witness_guards_agree_with_same_on_grid_specimen():
    law,fixture,config = InterfaceLaw(area=.01),Fixture(),settings()
    controllers = [FiniteGuard(law,fixture,config,.00001) for _ in range(2)]
    energy,prep = controllers[0].model.parameters(controllers[0].unit[[54]])[0]
    _,maximum = equilibrium_cycle(0.,energy,prep,law,fixture)
    actions,y = [0.,config['weak_drive_m']],[]
    for command in actions:
        value,maximum = equilibrium_cycle(maximum,energy,command,law,fixture)
        y.append(float(value)+.00003)
    reference = float(maximum)
    for turn in (0,1):
        choices = []
        for controller,policy in zip(controllers,('full_guard','witness_guard')):
            choice,record = controller.plan(actions,y,policy,turn)
            assert not record['chosen_action_finite_sample_forbidden']
            assert controller.previous_mask[54]
            choices.append(choice)
        assert choices[0] == choices[1]
        observed,maximum = equilibrium_cycle(maximum,energy,choices[0],law,fixture)
        assert damage(maximum,energy,law)-damage(reference,energy,law) <= .02+1e-12
        actions.append(choices[0]);y.append(float(observed)+.00003)


def test_empty_set_abstains_and_mutated_history_is_rejected():
    controller = FiniteGuard(InterfaceLaw(area=.01),Fixture(),settings(),.00001)
    choice,record = controller.plan([0.,.001],[.1,.1],'full_guard',0)
    assert choice == 0 and record['reason']=='empty_observation_set_abstention'
    assert record['finite_sample_maximum_damage_by_action'] is None
    with pytest.raises(ValueError,match='continue'):
        controller.plan([0.,.001,0.],[0.,.1,.1],'full_guard',1)


def test_full_audit_does_not_feed_back_into_top8_selection(monkeypatch):
    import wmfracture.damage_guard as module
    config,law,fixture = settings(),InterfaceLaw(area=.01),Fixture()
    first = FiniteGuard(law,fixture,config,.001)
    action,record = first.plan([0.,.001],[.00003,.0005],'top8_guard',0)
    original = module.action_risks
    def corrupted_audit(maximum,baseline,energy,actions,law,fixture):
        risks,failures = original(maximum,baseline,energy,actions,law,fixture)
        if len(energy)>config['subset_size']:
            risks = np.full_like(risks,.9)
        return risks,failures
    monkeypatch.setattr(module,'action_risks',corrupted_audit)
    second = FiniteGuard(law,fixture,config,.001)
    same,changed_record = second.plan([0.,.001],[.00003,.0005],'top8_guard',0)
    assert action == same
    assert record['finite_sample_maximum_damage_by_action'] != changed_record['finite_sample_maximum_damage_by_action']


def test_forecast_needs_no_private_trace_or_true_parameters(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from finite_damage_guard import forecast
    public = [{'id':'e','sigma_m':.00002,'actions_m':[0.,.001,.0012],
        'observed_gaps_m':[.00003,.00055,.00065]}]
    (tmp_path/'public-episodes.json').write_text(json.dumps(public))
    (tmp_path/'decisions.json').write_text('[]')
    seals = forecast(tmp_path,settings(),{'future_drives_m':[.0031]},InterfaceLaw(area=.01),Fixture())
    assert 'forecasts.json' in seals and not (tmp_path/'private-traces.json').exists()
    assert (tmp_path/'forecast-cost.json').exists()


def test_finer_audit_needs_only_public_decision_prefix(tmp_path,monkeypatch):
    import hashlib
    import json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from finite_damage_guard import fine_grid_audit
    decision = {'episode_id':'e','round':0,'policy':'full_guard','sigma_m':.00002,
        'public_prefix':{'actions_m':[0.,.001],'observed_gaps_m':[.00003,.00055]},
        'empty_set':False,'action_m':0.}
    decision['decision_sha256'] = hashlib.sha256(json.dumps(decision,sort_keys=True).encode()).hexdigest()
    (tmp_path/'decisions.json').write_text(json.dumps([decision]))
    audit = fine_grid_audit(tmp_path,settings(),InterfaceLaw(area=.01),Fixture())
    assert audit['fine_side'] == 161 and len(audit['records'])==1
    assert not (tmp_path/'private-traces.json').exists()
