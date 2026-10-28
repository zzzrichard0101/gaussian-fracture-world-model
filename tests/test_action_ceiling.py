import numpy as np
import pytest

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import PrefixModel,equilibrium_cycle,damage
from wmfracture.damage_guard import action_risks,admissibility,information_scores,FiniteGuard
from wmfracture.action_ceiling import finite_action_limit,proxy_optimal_action,noiseless_history_manifold


def physics():
    law,fixture = InterfaceLaw(area=.01),Fixture()
    model = PrefixModel(law,fixture,[[1.2,3.],[.0009,.0023]])
    unit = np.random.default_rng(7).uniform(.1,.9,(17,2))
    prefix,maximum = model.predict(unit,[0.,.001])
    return law,fixture,model.parameters(unit)[:,0],prefix,maximum


@pytest.mark.parametrize('cap',[0.,.02,.10])
def test_analytical_action_limit_matches_independent_risk_bisection(cap):
    law,fixture,energy,_,maximum = physics()
    limit,valid = finite_action_limit(maximum,maximum,energy,cap,.0035,law,fixture)
    assert valid
    low,high = 0.,.0035
    for _ in range(60):
        risks,failures = action_risks(maximum,maximum,energy,[(low+high)/2],law,fixture)
        if admissibility(risks,failures,cap)[0]:
            low = (low+high)/2
        else:
            high = (low+high)/2
    assert limit == pytest.approx(low,abs=2e-11)
    risks,failures = action_risks(maximum,maximum,energy,[limit],law,fixture)
    assert admissibility(risks,failures,cap)[0]


def test_empty_or_already_violating_set_is_not_unrestricted_safety():
    law,fixture,energy,_,maximum = physics()
    assert finite_action_limit(np.array([]),np.array([]),np.array([]),.02,.003,law,fixture)==(0.,False)
    _,after = equilibrium_cycle(maximum,energy,.0035,law,fixture)
    limit,valid = finite_action_limit(after,maximum,energy,.02,.0035,law,fixture)
    assert not valid and limit==0.


@pytest.mark.parametrize('tau',[0.,.00005])
def test_affine_sweep_matches_dense_action_search_including_softening(tau):
    law,fixture,energy,prefix,maximum = physics()
    weights = np.random.default_rng(11).random(len(energy));weights /= weights.sum()
    observed = [0.00003,.00055];sigma=.00002
    solution = proxy_optimal_action(prefix,observed,sigma,tau,weights,maximum,energy,.0035,law,fixture)
    dense = np.linspace(0.,.0035,3001)
    scores = information_scores(prefix,observed,sigma,tau,weights,maximum,energy,dense,law,fixture)
    assert solution['score'] >= float(scores.max())-1e-10
    direct = information_scores(prefix,observed,sigma,tau,weights,maximum,energy,[solution['action_m']],law,fixture)[0]
    assert solution['score'] == direct


def test_zero_action_domain_retains_shared_bias_observation_score():
    law,fixture,energy,prefix,maximum = physics()
    weights = np.ones(len(energy))/len(energy)
    solution = proxy_optimal_action(prefix,[.00003,.00055],.00002,.00005,weights,maximum,energy,0.,law,fixture)
    expected = information_scores(prefix,[.00003,.00055],.00002,.00005,weights,maximum,energy,[0.],law,fixture)[0]
    assert solution['action_m']==0. and solution['score']==expected


def test_continuous_proxy_policy_dominates_coarse_proxy_at_same_prefix():
    from test_damage_guard import settings
    config,law,fixture = settings(),InterfaceLaw(area=.01),Fixture()
    controllers = [FiniteGuard(law,fixture,config,.00002) for _ in range(2)]
    records = []
    for controller,policy in zip(controllers,('full_guard','continuous_guard')):
        _,record = controller.plan([0.,.001],[.00003,.0005856],policy,0)
        assert not record['chosen_action_finite_sample_forbidden']
        records.append(record)
    coarse = max(np.array(records[0]['information_scores'])[records[0]['admissible_actions']])
    fine = max(np.array(records[1]['information_scores'])[records[1]['admissible_actions']])
    assert fine >= coarse-1e-9
    assert records[1]['continuous_action_diagnostic']['finite_limit_valid']


def test_affine_sweep_finds_interior_mode_switch_not_only_largest_input():
    law,fixture = InterfaceLaw(area=.01),Fixture()
    energy = np.array([1.37,2.63]);ratio=.08
    final = 2*energy/law.peak_traction;onset=law.initiation_opening
    maximum = onset*final/(onset+ratio*(final-onset))
    gap,_ = equilibrium_cycle(maximum,energy,.001,law,fixture)
    prefix = np.column_stack((np.zeros(2),gap))
    result = proxy_optimal_action(prefix,[0.,gap[0]],.00002,0.,[.5,.5],maximum,energy,.006,law,fixture)
    assert 0 < result['action_m'] < .006
    assert result['action_m'] == pytest.approx(final[0],abs=1e-12)


def test_exact_unloading_curve_bound_covers_continuous_energy_interval():
    law,fixture = InterfaceLaw(area=.01),Fixture()
    result = noiseless_history_manifold(.08,[1.2,3.],[.0009,.0023],.001,.02,.00185,law,fixture)
    assert result['valid'] and result['energy_interval']==[1.2,3.]
    energy = np.linspace(*result['energy_interval'],10001)
    final = 2*energy/law.peak_traction;onset = law.initiation_opening
    maximum = onset*final/(onset+.08*(final-onset))
    risks,failures = action_risks(maximum,maximum,energy,[result['one_step_limit_m']],law,fixture)
    assert admissibility(risks,failures,.02)[0]
    assert risks[0,0] == pytest.approx(.02,abs=1e-10)


def test_noiseless_manifold_respects_preparation_command_bounds():
    law,fixture = InterfaceLaw(area=.01),Fixture()
    result = noiseless_history_manifold(.08,[1.2,3.],[.00165,.0018],.001,.02,.00185,law,fixture)
    assert result['valid']
    low,high = result['energy_interval']
    assert 1.2 < low < high < 3.
    energies = np.array([low,high]);final = 2*energies/law.peak_traction;onset = law.initiation_opening
    maximum = onset*final/(onset+.08*(final-onset))
    commands = maximum*(1+law.area*law.tensile_stiffness*.08/fixture.spring)
    np.testing.assert_allclose(commands,[.00165,.0018],atol=1e-12)
