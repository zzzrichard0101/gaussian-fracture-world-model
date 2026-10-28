from fractions import Fraction

import pytest

from wmfracture.verified_policy import filter_proposal
from wmfracture.continuous_risk import ContinuousRisk,replay_proof,rational


def inputs():
    return {'parameter_bounds':[[1.2,3.],[.0009,.0023]],'spring':10000.,
        'law':{'area':.01,'tensile_stiffness':1e7,'peak_traction':1000.},
        'actions_m':[0.,.001],'observed_gaps_m':[0.,.0005],'reference_count':2,
        'noise_radius_m':.00006,'bias_bound_m':.00015,'damage_cap':.02}


def test_filter_executes_only_a_context_bound_allow_after_unsafe_proposal():
    data=inputs();result=filter_proposal(data,.00185,lambda a:[(2.,.0016)],max_cells=511,backtracking_steps=4)
    assert not result['stopped'] and result['action_m']<.00185
    assert result['proof']['decision']=='allow'
    assert replay_proof(result['proof'],expected_inputs=data,expected_action=result['action_m'])
    assert result['attempts'][0]['decision']=='reject'


def test_empty_support_stops_without_unverified_zero_fallback():
    data=inputs();data['observed_gaps_m']=[.1,.1]
    result=filter_proposal(data,.00185,lambda a:[],max_cells=31)
    assert result['stopped'] and result['action_m'] is None and result['proof'] is None


def test_budget_abstention_is_recorded_and_does_not_authorize_proposal():
    data=inputs();result=filter_proposal(data,.0012,lambda a:[(2.,.0016)],max_cells=1,backtracking_steps=0)
    assert result['attempts'][0]['decision']=='abstain'
    assert result['action_m']==0. and result['proof']['decision']=='allow'


def test_invalid_backtracking_budget_rejected():
    with pytest.raises(ValueError):
        filter_proposal(inputs(),.0012,lambda a:[],backtracking_steps=-1)
