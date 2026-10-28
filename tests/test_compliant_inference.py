from dataclasses import replace

import numpy as np
import pytest

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture, DriveState, cycle
from wmfracture.compliant_inference import (PrefixModel, equilibrium_cycle, damage,
    normalize, grid, reflect, systematic_resample, ParticleFilter, log_likelihood)


def test_quasistatic_cycle_matches_inertial_history_and_softening():
    law, fixture = InterfaceLaw(area=.01), Fixture()
    maximum, state = 0., DriveState()
    for drive in (.0018, .001, .0023, .001):
        expected, maximum = equilibrium_cycle(maximum, law.fracture_energy, drive, law, fixture)
        rows, _ = cycle(state, law, fixture, drive, 4e-5, ramp=1., hold=.25, rest=.5)
        held = [r for r in rows if r['segment'] == 'hold'][-1]
        assert held['gap'] == pytest.approx(float(expected), abs=1e-7)
        assert state.maximum_opening == pytest.approx(float(maximum), abs=1e-7)


def test_quasistatic_rejects_unstable_branch_and_preserves_failure_on_unload():
    law, fixture = InterfaceLaw(area=.01), Fixture()
    with pytest.raises(ValueError, match='unsupported'):
        equilibrium_cycle(.001, 2., .002, law, replace(fixture, spring=1000.))
    _, maximum = equilibrium_cycle(0., 2., .005, law, fixture)
    gap, after = equilibrium_cycle(maximum, 2., .001, law, fixture)
    assert gap == pytest.approx(.001) and after == maximum
    assert damage(after, 2., law) == 1.


def test_batch_prefix_matches_independent_material_replay():
    model = PrefixModel(InterfaceLaw(area=.01), Fixture(), [[1.2, 3.], [.0009, .0023]])
    unit = np.array([[.21, .87], [.73, .29]])
    values, maxima = model.predict(unit, [.001, .00165])
    for index, point in enumerate(unit):
        single, maximum = model.predict(point[None], [.001, .00165])
        np.testing.assert_array_equal(single[0], values[index])
        assert maximum[0] == maxima[index]
    assert model.cost['parameter_prefix_evaluations'] == 4
    with pytest.raises(ValueError):
        model.predict(np.array([[1.1, .5]]), [.001])


def test_resampling_normalization_and_reflection_are_bounded():
    weights = normalize(np.array([-10000., -10001., -10002.]))
    assert weights.sum() == pytest.approx(1.)
    np.testing.assert_array_equal(systematic_resample(np.array([0., 1., 0.]), np.random.default_rng(1)), [1, 1, 1])
    result = reflect(np.array([-4.2, -.1, .4, 1.1, 5.2]))
    assert (result >= 0).all() and (result <= 1).all()


def test_no_resampling_smc_does_not_count_old_observations_twice():
    model = PrefixModel(InterfaceLaw(area=.01), Fixture(), [[1.2, 3.], [.0009, .0023]])
    smc = ParticleFilter(model, 128, 31, move_steps=0)
    initial = smc.unit.copy()
    observed = [.0005, .0009]
    first = smc.update([.001], observed[:1], .01)
    second = smc.update([.001, .00165], observed, .01)
    assert not first['resampled'] and not second['resampled']
    predictions, _ = model.predict(initial, [.001, .00165])
    expected = normalize(log_likelihood(predictions, observed, .01))
    np.testing.assert_allclose(smc.weights, expected, atol=1e-14)


def test_moved_particles_reconstruct_their_own_complete_physical_history():
    model = PrefixModel(InterfaceLaw(area=.01), Fixture(), [[1.2, 3.], [.0009, .0023]])
    smc = ParticleFilter(model, 128, 9, move_steps=3)
    original = smc.unit.copy()
    trace = smc.update([.001, .00165], [.00055, .00097], .00005)
    _, replayed = model.predict(smc.unit, [.001, .00165])
    np.testing.assert_array_equal(smc.maximum, replayed)
    assert trace['accepted_moves'] > 0
    assert not np.array_equal(original, smc.unit)
    assert (smc.unit >= 0).all() and (smc.unit <= 1).all()


def test_preparation_command_not_hidden_damage_is_the_unknown_parameter():
    model = PrefixModel(InterfaceLaw(area=.01), Fixture(), [[1.2, 3.], [.0009, .0023]])
    points = grid(3)
    predictions, maximum = model.predict(points, [.001, .00185])
    assert predictions.shape == (9, 2)
    assert np.isfinite(predictions).all()
    _, first = model.predict(points, [.001])
    assert (maximum >= first).all()


def test_fit_and_forecast_run_without_private_labels_or_future_truth(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from synthetic_filter_benchmark import fit, forecast
    config = {'parameter_bounds':[[1.2, 3.], [.0009, .0023]], 'grid_sides':[7],
        'protocols':{'fixed_probe':[.001, .00165]}, 'seed':41, 'hard_sigma_multiple':3.,
        'smc':{'particles':16, 'replicates':1, 'move_steps':2, 'move_scale':.08, 'global_probability':.1}}
    episodes = [{'id':'episode-0', 'case_id':'hidden', 'protocol':'fixed_probe',
        'actions_m':[.001, .00165], 'observed_gaps_m':[.1, .1], 'sigma_m':.00002}]
    (tmp_path/'observations.json').write_text(json.dumps(episodes))
    law, fixture = InterfaceLaw(area=.01), Fixture()
    fit(tmp_path, config, law, fixture)
    # There is deliberately no private-cases file and no future plan at fit time.
    packets_before = (tmp_path/'fit-packets.json').read_bytes()
    seals = forecast(tmp_path, {**config, 'future_drives_m':[.0031]}, law, fixture)
    assert set(seals) == {'posteriors','packets','forecasts'}
    assert (tmp_path/'fit-packets.json').read_bytes() == packets_before
    forecasts = json.loads((tmp_path/'forecasts.json').read_text())
    hard = [r for r in forecasts if r['method']=='grid7_hard']
    assert hard and all(r['empty'] and r['futures']==[] for r in hard)


def test_hard_candidate_set_does_not_claim_probabilistic_intervals(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from synthetic_filter_benchmark import forecast
    points = np.array([[.2, .4], [.7, .6]])
    np.savez(tmp_path/'posteriors.npz', unit=points, weights=np.array([.5, .5]))
    (tmp_path/'observations.json').write_text(json.dumps([{'id':'e', 'actions_m':[.001]}]))
    (tmp_path/'fit-packets.json').write_text(json.dumps([{'episode_id':'e','stage':1,
        'method':'grid7_hard','unit_key':'unit','weight_key':'weights'}]))
    forecast(tmp_path, {'parameter_bounds':[[1.2,3.],[.0009,.0023]],'future_drives_m':[.0031]},
        InterfaceLaw(area=.01), Fixture())
    future = json.loads((tmp_path/'forecasts.json').read_text())[0]['futures'][0]
    assert future['q025_gap_m'] is None and future['q975_gap_m'] is None
    assert future['separation_mass'] is None
    assert future['support_min_gap_m'] <= future['support_max_gap_m']
