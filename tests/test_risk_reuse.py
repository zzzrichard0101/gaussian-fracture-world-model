from copy import deepcopy

import pytest

from wmfracture.continuous_risk import ContinuousRisk, rational, replay_proof
from wmfracture.verified_policy import RiskReuseSession, filter_proposal


def inputs():
    return {'parameter_bounds':[[1.2,3.],[.0009,.0023]],'spring':10000.,
        'law':{'area':.01,'tensile_stiffness':1e7,'peak_traction':1000.},
        'actions_m':[0.,.001],'observed_gaps_m':[0.,.0005],'reference_count':2,
        'noise_radius_m':.00006,'bias_bound_m':.00015,'damage_cap':.02}


def semantic(proof):
    return {k:v for k,v in proof.items() if k not in ('point_evaluations','reuse_stats')}


def test_same_cover_witness_and_decision_with_fewer_constraint_evaluations():
    cold=ContinuousRisk(inputs());cached=ContinuousRisk(inputs(),reuse_observations=True)
    counts=[0,0]
    for action in (.0012,.00185,.0012,.0011):
        proofs=[v.verify(action,max_cells=511,warm_points=[(2.,.0016)]) for v in (cold,cached)]
        assert semantic(proofs[0])==semantic(proofs[1])
        for i,p in enumerate(proofs):
            counts[i]+=p['reuse_stats']['box_constraints']
            assert replay_proof(p)  # Always works with a fresh standalone checker.
    assert counts[1]<counts[0]


def test_append_rechecks_surviving_support_and_current_history_risk():
    data=inputs();cached=ContinuousRisk(data,reuse_observations=True)
    point=(rational(2.),rational(.0016))
    assert cached.point_support(*point)[0]
    cached.assess(cached.box,rational(.0012))
    updated=deepcopy(data);updated['actions_m'].append(.00185);updated['observed_gaps_m'].append(.1)
    cached.append_observations(updated);cold=ContinuousRisk(updated)
    assert not cached.point_support(*point)[0]
    assert cached.point_support(*point)==cold.point_support(*point)
    assert cached.assess(cached.box,rational(.0012))==cold.assess(cold.box,rational(.0012))
    assert cached.point_risk(*point,rational(0.))==cold.point_risk(*point,rational(0.))
    assert cached.reuse_stats['point_extensions']>0 and cached.reuse_stats['box_extensions']>0


@pytest.mark.parametrize('field,value',[
    ('noise_radius_m',.001),('bias_bound_m',.001),('damage_cap',.1),
    ('spring',11000.),('reference_count',3),('parameter_bounds',[[1.3,3.],[.0009,.0023]])])
def test_changed_assumptions_cannot_reuse_cache(field,value):
    session=RiskReuseSession();session.context(inputs())
    changed=inputs();changed['actions_m'].append(.001);changed['observed_gaps_m'].append(.0005)
    changed[field]=value
    with pytest.raises(ValueError):session.context(changed)
    assert session.search.inputs==inputs() and session.checker.inputs==inputs()


def test_prefix_cannot_be_rewritten_and_search_does_not_supply_checker_bounds():
    session=RiskReuseSession();search,checker=session.context(inputs())
    proof=search.verify(.0012,max_cells=511,warm_points=[(2.,.0016)])
    assert checker._observation_cache=={} and search._observation_cache
    assert replay_proof(proof,verifier=checker)
    forged=deepcopy(proof);forged['upper_damage']=.9
    with pytest.raises(ValueError):replay_proof(forged,verifier=checker)
    changed=inputs();changed['actions_m']=[0.,.0011,.0012];changed['observed_gaps_m'].append(.0006)
    with pytest.raises(ValueError):session.context(changed)
    with pytest.raises(ValueError):replay_proof(proof,verifier=ContinuousRisk(changed))


def test_two_step_filter_actions_and_proofs_equal_cold_replay():
    data=inputs();session=RiskReuseSession()
    for _ in range(2):
        args=(data,.00185,lambda a:[(2.,.0016)])
        cold=filter_proposal(*args,max_cells=511,backtracking_steps=4)
        reused=filter_proposal(*args,max_cells=511,backtracking_steps=4,reuse_session=session)
        assert cold['action_m']==reused['action_m']
        assert semantic(cold['proof'])==semantic(reused['proof'])
        assert replay_proof(reused['proof'],expected_inputs=data,expected_action=reused['action_m'])
        action=cold['action_m'];model=ContinuousRisk(data)
        observed=float(model.gap(rational(2.),rational(.0016),rational(action),max(model.current_peak,rational(action))))
        data=deepcopy(data);data['actions_m'].append(action);data['observed_gaps_m'].append(observed)
