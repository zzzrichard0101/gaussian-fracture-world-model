from copy import deepcopy

import numpy as np
import pytest

from wmfracture.spatial_observation import PhysicalCatalog
from wmfracture.spatial_risk import point_evidence, verify, replay


def query(local=True):
    point = [2., .0018, .0018]
    bank = PhysicalCatalog([point], [[0., 0.], [.0008, .0008]], [[.0031, .0031]])
    observed = [bank.observations((0,), [-.04, .04])[0].tolist(), bank.observations((0, 1), [-.04, .04])[0].tolist()]
    return {'parameter_bounds': [[1.99999, 2.00001], [.00179999, .00180001], [.00179999, .00180001]] if local else [[1.3, 2.8], [.0008, .0018], [.0008, .0018]],
        'actions_m': [[0., 0.], [.0008, .0008]], 'observed_m': observed, 'positions_m': [-.04, .04],
        'proposed_m': [.00145, .0004], 'noise_radius_m': .00006, 'bias_bound_m': .00015,
        'damage_cap': .02, 'increments': 8}


def test_allow_requires_both_cover_and_nonempty_validated_support():
    data = query()
    result = verify(data, max_cells=1, warm_points=[[2., .0018, .0018]])
    assert result['decision'] == 'allow' and replay(result, data)
    absent = verify(data, max_cells=1)
    assert absent['decision'] == 'abstain'


def test_budget_abstention_is_not_safety_or_inconsistency():
    data = query(False)
    result = verify(data, max_cells=3, warm_points=[[2., .0018, .0018]])
    assert result['decision'] == 'abstain' and len(result['nodes']) == 3
    assert replay(result)


def test_bad_observations_prove_empty_domain_instead_of_allowing_it():
    data = query(False); data['observed_m'] = [[.01, .01], [.01, .01]]
    result = verify(data, max_cells=1)
    assert result['decision'] == 'inconsistent' and replay(result)


def test_interval_point_feasibility_is_sufficient_and_rejection_has_a_lower_bound():
    data = query(False); point = [1.3803881930187345, .0014446525566279888, .001244484293460846]
    bank = PhysicalCatalog([point], [[0., 0.], [.0008, .0008]], [[.0031, .0031]])
    data['observed_m'] = [bank.observations((0,), [-.04, .04])[0].tolist(), bank.observations((0, 1), [-.04, .04])[0].tolist()]
    data['proposed_m'] = [.0004, .00145]
    result = verify(data, max_cells=1, warm_points=[point])
    assert result['decision'] == 'reject' and result['points'][0]['risk_lower'] > .02
    assert replay(result)
    bad = deepcopy(result); bad['points'][0]['risk_lower'] = 0.
    with pytest.raises(ValueError): replay(bad)


def test_checker_rejects_partition_gaps_and_forged_safety():
    result = verify(query(False), max_cells=3)
    bad = deepcopy(result); bad['nodes'][1]['box'][0][1] += .01
    with pytest.raises(ValueError): replay(bad)
    bad = deepcopy(result); bad['decision'] = 'allow'
    with pytest.raises(ValueError): replay(bad)


def test_witness_must_be_in_domain_and_public_inputs_must_match():
    data = query()
    with pytest.raises(ValueError): point_evidence(data, [1., .001, .001])
    result = verify(data, max_cells=1)
    modified = deepcopy(data); modified['proposed_m'] = [0., 0.]
    with pytest.raises(ValueError): replay(result, modified)
