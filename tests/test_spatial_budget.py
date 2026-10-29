import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import qmc

from wmfracture.spatial_observation import PhysicalCatalog
from wmfracture.spatial_transport import DefensiveMixture, fit_proposals
from wmfracture.spatial_budget import (path_endpoint_cost, main_allowance, physical_pilot_proposal,
    equal_hybrid, select_physics_draws, cutoff_for_allowance)


def test_endpoint_budget_matches_actual_preparation_prefix_and_future_cost():
    actions = [[0., 0.], [.0008, .0008]]
    futures = [[0., 0.], [.002, .002], [.003, .001]]
    bank = PhysicalCatalog([[2., .0012, .0014]], actions, futures, 3)
    bank.future((0, 1, 0))
    assert bank.physics.calls == path_endpoint_cost(actions, futures, [0, 1, 0], 3) == 24
    assert path_endpoint_cost(actions, futures, [0, 1, 0], 3, False) == 12


def test_main_allowance_charges_setup_and_never_exceeds_budget():
    assert main_allowance(1000, 150, 128) == 6
    assert 150+6*128 <= 1000 < 150+7*128
    assert main_allowance(128, 128, 128) == 0
    with pytest.raises(ValueError): main_allowance(100, 101, 10)
    with pytest.raises(ValueError): main_allowance(100, 0, 0)


def test_hybrid_density_is_equal_mixture_of_two_full_defensive_densities():
    broad = DefensiveMixture([[.9,.8,.7]], [np.eye(3)*.2**2], .2)
    narrow = DefensiveMixture([[.6,.4,.3]], [np.eye(3)*.03**2], .2)
    hybrid = equal_hybrid(broad, narrow)
    points = np.array([[.5,.4,.2],[1.2,.8,.6]])
    np.testing.assert_allclose(np.exp(hybrid.log_density(points)),
        .5*np.exp(broad.log_density(points))+.5*np.exp(narrow.log_density(points)))


def test_whole_mixture_conditioning_preserves_normalized_uniform_target():
    proposal = DefensiveMixture([[.99,.99,.99]], [np.eye(3)*.12**2], .2)
    unit = qmc.Sobol(4,scramble=True,seed=77).random_base2(16)
    points, logq, inside = select_physics_draws(proposal, unit, 8192)
    assert inside.sum() == 8192 and inside[-1] and len(points) > 12000
    weights = np.where(inside, np.exp(-logq), 0.)
    np.testing.assert_allclose((weights[:,None]*points).sum(axis=0)/weights.sum(), .5, atol=.008)
    assert (weights*(points[:,0]<.2)).sum()/weights.sum() == pytest.approx(.2, abs=.008)


def test_nested_budget_cutoff_uses_only_geometry_and_preserves_prefix():
    unit = qmc.Sobol(4,scramble=True,seed=78).random_base2(10)
    proposal = DefensiveMixture([[.98,.98,.98]], [np.eye(3)*.1**2], .2)
    points, logq, inside = select_physics_draws(proposal,unit,128,chunk=32)
    end = cutoff_for_allowance(inside,64)
    small, small_q, small_mask = select_physics_draws(proposal,unit,64,chunk=128)
    np.testing.assert_array_equal(points[:end],small)
    np.testing.assert_array_equal(logq[:end],small_q)
    np.testing.assert_array_equal(inside[:end],small_mask)
    with pytest.raises(ValueError): select_physics_draws(proposal,unit[:4],128)


def test_pilot_only_physical_proposal_matches_old_baseline_without_projection():
    class LinearForward:
        def evaluate(self, unit, prefix, future=False):
            prediction = np.zeros((len(unit),len(prefix),2))
            prediction[:,1,0] = .001*(unit[:,1]+.2*unit[:,0])
            prediction[:,1,1] = .001*(unit[:,2]-.3*unit[:,0])
            return prediction,None
    config = json.loads((Path(__file__).resolve().parents[1]/'config/spatial-transport.json').read_text())
    config['charts'] = 4
    unit = qmc.Sobol(3,scramble=True,seed=79).random_base2(7)
    model = LinearForward(); prediction,_ = model.evaluate(unit,[0,1]); observed = prediction[0]
    settings = {'bias_sigma_m':.00005}
    old,_ = fit_proposals(model,unit,prediction,[0,1],observed,1e-6,settings,config)
    new = physical_pilot_proposal(unit,prediction,observed,1e-6,settings,config)
    np.testing.assert_array_equal(old['physical_mixture'].centers,new.centers)
    np.testing.assert_array_equal(old['physical_mixture'].covariances,new.covariances)
