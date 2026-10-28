import numpy as np
import pytest
from scipy.linalg import solve

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import PrefixModel, ParticleFilter, normalize, log_likelihood
from wmfracture.measurement_bias import GaussianObservation


def test_shared_score_matches_direct_correlated_covariance_solve():
    prediction = np.array([[.0002,.0005,.0008], [.0004,.0004,.0009]])
    observed = np.array([.00024,.00053,.00083]); sigma, tau = .00002, .00005
    covariance = sigma**2*np.eye(3)+tau**2*np.ones((3,3))
    residual = observed-prediction
    expected = np.array([-.5*r@solve(covariance,r,assume_a='pos') for r in residual])
    np.testing.assert_allclose(GaussianObservation('shared',tau)(prediction,observed,sigma), expected, atol=1e-13)


def test_zero_bias_recovers_original_independent_model():
    prediction = np.array([[0.,.001], [.0003,.0011]])
    observed = np.array([.0001,.001])
    np.testing.assert_array_equal(GaussianObservation('shared',0.)(prediction,observed,.00002),
        log_likelihood(prediction, observed, .00002))


def test_repeated_offset_does_not_average_away_like_independent_noise():
    model = GaussianObservation('shared',.00005)
    one = model(np.zeros((1,1)),np.full(1,.00003),.000005)[0]
    hundred = model(np.zeros((1,100)),np.full(100,.00003),.000005)[0]
    assert abs(hundred/one) < 1.02
    independent = GaussianObservation('inflated',.00005)
    assert independent(np.zeros((1,100)),np.full(100,.00003),.000005)[0] == pytest.approx(100*one)


def test_bias_posterior_matches_numerical_integration_and_mixture_variance():
    model = GaussianObservation('shared',.00005)
    prediction = np.array([[0.,0.], [.00001,.00001]])
    observed, sigma = np.array([.00003,.00004]), .00002
    weights = np.array([.3,.7])
    exact = model.bias_moments(prediction, observed, sigma, weights)
    b = np.linspace(-.0004,.0004,20001)
    first, second = [], []
    for row in prediction:
        density = np.exp(-.5*(b/.00005)**2-.5*np.sum(((observed-row-b[:,None])/sigma)**2,axis=1))
        density /= np.trapz(density,b)
        first.append(np.trapz(density*b,b)); second.append(np.trapz(density*b*b,b))
    mean = weights@first
    assert exact['mean_m'] == pytest.approx(mean,abs=1e-12)
    assert exact['std_m'] == pytest.approx(np.sqrt(weights@second-mean**2),abs=1e-12)


def test_correlated_smc_increments_match_batch_posterior_without_double_counting():
    model = PrefixModel(InterfaceLaw(area=.01),Fixture(),[[1.2,3.],[.0009,.0023]])
    likelihood = GaussianObservation('shared',.001)
    smc = ParticleFilter(model,128,17,move_steps=0,likelihood=likelihood)
    initial = smc.unit.copy(); y = [.0005,.0009,.00105]; actions = [.001,.00165,.00185]
    for stage in (1,2,3):
        trace = smc.update(actions[:stage],y[:stage],.01)
        assert not trace['resampled']
    predictions, _ = model.predict(initial,actions)
    np.testing.assert_allclose(smc.weights,normalize(likelihood(predictions,y,.01)),atol=1e-14)
    with pytest.raises(ValueError,match='unchanged prefix'):
        smc.update(actions+[.002], [0.,.0009,.00105,.0012], .01)


def test_initial_multi_observation_update_uses_all_frames_including_rest():
    model = PrefixModel(InterfaceLaw(area=.01),Fixture(),[[1.2,3.],[.0009,.0023]])
    likelihood = GaussianObservation('shared',.001)
    smc = ParticleFilter(model,64,73,move_steps=0,likelihood=likelihood)
    original = smc.unit.copy(); y = [.00003,.0005]; actions = [0.,.001]
    trace = smc.update(actions,y,.01)
    assert not trace['resampled']
    predicted, _ = model.predict(original,actions)
    np.testing.assert_allclose(smc.weights,normalize(likelihood(predicted,y,.01)),atol=1e-14)
    np.testing.assert_array_equal(smc.predictions,predicted)


def test_zero_load_calibration_does_not_change_physical_preparation_history():
    model = PrefixModel(InterfaceLaw(area=.01),Fixture(),[[1.2,3.],[.0009,.0023]])
    points = np.array([[.1,.9],[.8,.2],[.4,.6]])
    ordinary, maximum = model.predict(points,[.001,.00165])
    calibrated, calibrated_maximum = model.predict(points,[0.,.001,.00165])
    np.testing.assert_array_equal(calibrated[:,0],np.zeros(3))
    np.testing.assert_array_equal(calibrated[:,1:],ordinary)
    np.testing.assert_array_equal(calibrated_maximum,maximum)


def test_bias_fit_with_rest_needs_no_truth_scenario_or_future_plan(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from bias_aware_filter import fit
    settings = {'grid_side':7,'seed':91,'observation_modes':['independent','inflated','shared'],
        'bias_sigma_m':.00005,'sensitivity_bias_sigma_m':[.00005],
        'additional_synthetic_zero_load_calibration':True}
    base = {'parameter_bounds':[[1.2,3.],[.0009,.0023]],
        'smc':{'particles':16,'replicates':1,'move_steps':2,'move_scale':.08,'global_probability':.1}}
    episode = {'id':'e','actions_m':[.001,.00165,.00185],
        'observed_gaps_m':[.00055,.00097,.0011],'sigma_m':.00002,'rest_gap_m':.00003}
    (tmp_path/'observations.json').write_text(json.dumps([episode]))
    fit(tmp_path,settings,base,InterfaceLaw(area=.01),Fixture())
    packets = json.loads((tmp_path/'fit-packets.json').read_text())
    rest = [p for p in packets if p['method'].endswith('_rest')]
    assert rest and all(p['observations_used']==p['stage']+1 for p in rest)
    assert all('bias_posterior' in p for p in rest)
    assert not (tmp_path/'private-cases.json').exists()


def test_physical_forecast_does_not_add_measurement_bias_to_true_gap(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from synthetic_filter_benchmark import forecast
    from wmfracture.compliant_inference import equilibrium_cycle
    law,fixture = InterfaceLaw(area=.01),Fixture()
    bounds = [[1.2,3.],[.0009,.0023]]; point = np.array([[.3,.7]])
    np.savez(tmp_path/'posteriors.npz',unit=point,weights=np.ones(1))
    (tmp_path/'observations.json').write_text(json.dumps([{'id':'e','actions_m':[.001]}]))
    (tmp_path/'fit-packets.json').write_text(json.dumps([{'episode_id':'e','stage':1,
        'method':'shared_rest','unit_key':'unit','weight_key':'weights','bias_posterior':{'mean_m':.01}}]))
    forecast(tmp_path,{'parameter_bounds':bounds,'future_drives_m':[.0031]},law,fixture)
    predicted = json.loads((tmp_path/'forecasts.json').read_text())[0]['futures'][0]['mean_gap_m']
    model = PrefixModel(law,fixture,bounds)
    _,maximum = model.predict(point,[.001]);energy = model.parameters(point)[:,0]
    expected,_ = equilibrium_cycle(maximum,energy,.0031,law,fixture)
    assert predicted == expected[0]


@pytest.mark.parametrize('sigma,tau',[(0.,.01),(.01,-1.),(float('nan'),.01)])
def test_invalid_noise_assumptions_rejected(sigma,tau):
    with pytest.raises(ValueError):
        GaussianObservation('shared',tau)(np.zeros((1,1)),np.zeros(1),sigma)
