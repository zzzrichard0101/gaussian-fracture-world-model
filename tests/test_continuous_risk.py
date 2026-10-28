from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import PrefixModel,equilibrium_cycle,damage
from wmfracture.continuous_risk import ContinuousRisk,rational,outward,replay_proof


def inputs():
    return {'parameter_bounds':[[1.2,3.],[.0009,.0023]],'spring':10000.,
        'law':{'area':.01,'tensile_stiffness':1e7,'peak_traction':1000.},
        'actions_m':[0.,.001],'observed_gaps_m':[0.,.0005],'reference_count':2,
        'noise_radius_m':.00006,'bias_bound_m':.00015,'damage_cap':.02}


def test_exact_remote_history_reduction_matches_existing_forward_solver():
    data = inputs();data['actions_m']=[0.,.001,.0017,.0027,.0005]
    data['observed_gaps_m']=[0.]*5
    verifier = ContinuousRisk(data)
    model = PrefixModel(InterfaceLaw(area=.01),Fixture(),[[1.2,3.],[.0009,.0034]])
    unit = np.random.default_rng(71).random((50,2))
    predicted,maximum = model.predict(unit,data['actions_m'])
    for index,(e,h) in enumerate(model.parameters(unit)):
        exact = [float(verifier.gap(rational(e),rational(h),a,p)) for a,p in zip(verifier.actions,verifier.peaks)]
        np.testing.assert_allclose(exact,predicted[index],atol=1e-14,rtol=1e-12)
        assert float(verifier.damage(rational(e),max(rational(h),max(verifier.actions)))) == pytest.approx(float(damage(maximum[index],e,model.law)),abs=1e-13)


def test_box_risk_and_observation_bounds_cover_interior_mode_switches():
    data = inputs();data['noise_radius_m']=.01
    verifier = ContinuousRisk(data);action=rational(.0027)
    rng = np.random.default_rng(13)
    for _ in range(12):
        erange = sorted(rng.uniform(1.2,3.,2));hrange=sorted(rng.uniform(.0009,.0023,2))
        box = tuple(tuple(map(rational,pair)) for pair in (erange,hrange))
        infeasible,upper,possible_failure = verifier.assess(box,action)
        assert not infeasible
        for e in np.linspace(*erange,7):
            for h in np.linspace(*hrange,7):
                risk,failed = verifier.point_risk(rational(e),rational(h),action)
                assert risk<=upper
                assert not failed or possible_failure


def test_coupled_history_bound_is_zero_when_both_paths_share_the_peak():
    data=inputs();data['noise_radius_m']=.01
    verifier=ContinuousRisk(data)
    box=((rational(1.2),rational(3.)),(rational(.0018),rational(.0023)))
    infeasible,upper,failure=verifier.assess(box,rational(.0015))
    assert not infeasible and upper==0 and not failure


def test_allow_has_complete_cover_and_a_verified_feasible_support_point():
    result=ContinuousRisk(inputs()).verify(.001,max_cells=1,warm_points=[(2.,.0016)])
    assert result['decision']=='allow' and result['upper_damage']==0
    assert replay_proof(result)


def test_reject_requires_a_feasible_violating_point_not_failed_search():
    result=ContinuousRisk(inputs()).verify(.00185,max_cells=127,warm_points=[(2.,.0016)])
    assert result['decision']=='reject' and result['lower_damage']>.02
    assert replay_proof(result)


def test_small_budget_abstains_instead_of_treating_samples_as_upper_bound():
    result=ContinuousRisk(inputs()).verify(.0012,max_cells=1)
    assert result['decision']=='abstain' and result['open_leaves']>0
    assert result['lower_damage']<=.02 < result['upper_damage']
    assert replay_proof(result)


def test_empty_observation_domain_is_not_reported_safe():
    data=inputs();data['observed_gaps_m']=[.1,.1]
    result=ContinuousRisk(data).verify(0.,max_cells=31)
    assert result['decision']=='inconsistent' and result['witness'] is None
    assert replay_proof(result)


def test_proof_checker_rejects_partition_gap_and_forged_bounds():
    result=ContinuousRisk(inputs()).verify(.0012,max_cells=1023,warm_points=[(2.,.0016)])
    assert result['decision']=='allow' and len(result['proof_nodes'])>1
    assert replay_proof(result)
    broken=deepcopy(result);broken['proof_nodes'][1]['box'][0][0]='0'
    with pytest.raises(ValueError,match='partition'):
        replay_proof(broken)
    broken=deepcopy(result);broken['upper_damage']=0.9
    with pytest.raises(ValueError,match='display'):
        replay_proof(broken)


def test_unsupported_unstable_mechanics_are_rejected():
    data=inputs();data['spring']=1000.
    with pytest.raises(ValueError,match='stable'):
        ContinuousRisk(data)


def test_outward_serialization_encloses_exact_fraction():
    value=Fraction(1,10)
    assert rational(outward(value,False))<=value<=rational(outward(value,True))


def test_input_snapshot_and_replay_are_bound_to_actual_context_and_action():
    data=inputs();snapshot=deepcopy(data);verifier=ContinuousRisk(data)
    data['observed_gaps_m'][1]=.1
    result=verifier.verify(.001,max_cells=1,warm_points=[(2.,.0016)])
    assert result['decision']=='allow'
    assert replay_proof(result,expected_inputs=snapshot,expected_action=.001)
    with pytest.raises(ValueError,match='context'):
        replay_proof(result,expected_inputs=data)
    with pytest.raises(ValueError,match='action'):
        replay_proof(result,expected_action=.00185)


def test_forged_safe_status_on_unresolved_cover_is_rejected():
    result=ContinuousRisk(inputs()).verify(.0012,max_cells=1)
    assert result['decision']=='abstain'
    result['decision']='allow'
    with pytest.raises(ValueError,match='complete'):
        replay_proof(result)


def test_near_separation_modes_do_not_allow_small_damage_new_failure():
    data=inputs();data['parameter_bounds']=[[1.2,1.3],[.002,.0022]]
    data['actions_m']=[0.,.001,.0027];data['observed_gaps_m']=[0.,.001,.0027]
    data['reference_count']=3;data['noise_radius_m']=.01
    # Already broken reference is not a NEW failure; same history has zero extra D.
    verifier=ContinuousRisk(data)
    result=verifier.verify(.003,max_cells=1,warm_points=[(1.25,.0021)])
    assert result['decision']=='allow' and replay_proof(result)
    data['reference_count']=2
    result=ContinuousRisk(data).verify(.003,max_cells=15,warm_points=[(1.25,.0021)])
    assert result['decision']=='reject' and result['witness']['new_failure']
    assert replay_proof(result)


def test_safe_risk_without_confirmed_nonempty_support_abstains():
    data=inputs();data['noise_radius_m']=.000001;data['observed_gaps_m']=[0.,.0008]
    result=ContinuousRisk(data).verify(0.,max_cells=1)
    assert result['decision']=='abstain' and result['witness'] is None
    assert result['upper_damage']==0. and replay_proof(result)


def test_submitted_bias_and_supported_model_kind_are_checked():
    result=ContinuousRisk(inputs()).verify(.001,max_cells=1,warm_points=[(2.,.0016)])
    result['witness']['bias']='1'
    with pytest.raises(ValueError,match='bias'):
        replay_proof(result)
    data=inputs();data['model_kind']='inertial_dynamic'
    with pytest.raises(ValueError,match='model kind'):
        ContinuousRisk(data)
