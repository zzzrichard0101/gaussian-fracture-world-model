from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import root

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import equilibrium_cycle
from wmfracture.spatial_fixture import SpatialFixture, three_zone_fixture


def test_elastic_asymmetric_response_has_contact_and_balanced_force_and_moment():
    model = three_zone_fixture(); state = model.virgin()
    q, maximum, audit = model.equilibrium([.0005, 0.], state.maximum_opening)
    assert abs(audit['force_residual_n']) < 1e-10
    assert abs(audit['moment_residual_nm']) < 1e-11
    assert 'contact' in audit['modes']
    assert q[0] > 0 and q[1] < 0
    assert np.all(model.damages(maximum) == 0)
    assert audit['minimum_tangent_eigenvalue_n_m'] >= model.minimum_tangent_bound


def test_center_loading_reduces_to_independent_scalar_model_through_failure_and_contact():
    model = three_zone_fixture(); state = model.virgin()
    law = InterfaceLaw(area=.01); fixture = Fixture(spring=10000.)
    maximum = 0.
    for command in [.0008, .0013, 0., .0018, .0012, .0048, 0., -.0001]:
        expected, maximum = equilibrium_cycle(maximum, law.fracture_energy, max(command, 0.), law, fixture)
        if command < 0:
            expected = fixture.spring*command/(fixture.spring+law.area*law.compressive_stiffness)
        model.advance(state, [command, command])
        assert state.q[0] == pytest.approx(float(expected), abs=2e-15)
        assert state.q[1] == pytest.approx(0., abs=2e-15)
        np.testing.assert_allclose(state.maximum_opening, maximum, atol=2e-15)
    assert np.all(model.damages(state.maximum_opening) == 1.)
    assert np.all(model.B@state.q < 0.)


def test_single_zone_center_response_also_matches_scalar_model():
    law = InterfaceLaw(area=.01)
    model = SpatialFixture([law], [0.], [-.04, .04], [5000., 5000.])
    q, maximum, _ = model.equilibrium([.0018, .0018], np.zeros(1))
    expected, previous = equilibrium_cycle(0., 2., .0018, law, Fixture())
    assert q[0] == pytest.approx(float(expected), abs=1e-15)
    assert maximum[0] == pytest.approx(float(previous), abs=1e-15)


def test_enumerated_equilibria_match_independent_nonlinear_root_with_retained_history():
    model = three_zone_fixture(); state = model.virgin()
    model.cycle(state, [.0018, .0008], 2e-5)
    rng = np.random.default_rng(20261030)
    for drive in rng.uniform(-.0003, .0032, (25, 2)):
        old = state.maximum_opening.copy()
        def residual(q):
            gaps = model.B@q
            forces = [law.area*law.traction(float(gap), float(max(h, gap)))
                      for law, gap, h in zip(model.laws, gaps, old)]
            return model.drive_stiffness@q+model.B.T@forces-model.C.T@(model.springs*drive)
        reference = root(residual, np.zeros(2), tol=1e-10)
        assert np.max(abs(residual(reference.x))) < 1e-8
        q, maximum, _ = model.equilibrium(drive, old)
        np.testing.assert_allclose(q, reference.x, atol=2e-12, rtol=1e-10)
        np.testing.assert_allclose(maximum, np.maximum(old, model.B@q), atol=1e-15)


def test_mirrored_preparation_and_probing_obey_spatial_symmetry():
    model = three_zone_fixture(); left, right = model.virgin(), model.virgin()
    for command in ([.0018, .0008], [0., 0.], [.0014, .0004], [0., 0.]):
        model.segment(left, command, 2e-5)
        model.segment(right, command[::-1], 2e-5)
        assert left.q[0] == pytest.approx(right.q[0], abs=1e-15)
        assert left.q[1] == pytest.approx(-right.q[1], abs=1e-15)
        np.testing.assert_allclose(left.maximum_opening, right.maximum_opening[::-1], atol=1e-14)
        assert left.actuator_work == pytest.approx(right.actuator_work, abs=1e-13)


def test_restart_copies_complete_state_and_sequential_probe_never_erases_history():
    model = three_zone_fixture(); state = model.virgin()
    model.cycle(state, [.0018, .0008], 2e-5)
    copy = state.clone(); old = state.maximum_opening.copy(); previous_step = state.step
    copy.maximum_opening[0] += 1e-8
    assert state.maximum_opening[0] == old[0]
    copy = state.clone()
    for drive in ([.0006, .0017], [0., 0.], [.0008, .0008]):
        model.advance(state, drive); model.advance(copy, drive)
        np.testing.assert_array_equal(state.q, copy.q)
        np.testing.assert_array_equal(state.maximum_opening, copy.maximum_opening)
        np.testing.assert_array_equal(state.drive, copy.drive)
        assert state.actuator_work == copy.actuator_work and state.step == copy.step
        assert np.all(state.maximum_opening >= old)
        old = state.maximum_opening.copy()
    assert state.step > previous_step


def test_work_dissipation_and_displacement_refinement_on_asymmetric_cycles():
    model = three_zone_fixture(); results = []
    for resolution in (16e-6, 8e-6, 4e-6):
        state = model.virgin(); previous = np.zeros(3)
        for peak in ([.00183, .00081], [.00061, .00172]):
            rows, _ = model.cycle(state, peak, resolution)
            for row in rows:
                current = np.asarray(row['maximum_opening_m'])
                assert np.all(current >= previous-1e-15)
                previous = current
            assert np.max(abs(state.q)) < 1e-14
        results.append((state.clone(), model.energies(state)))
    np.testing.assert_allclose(results[-1][0].maximum_opening, results[-2][0].maximum_opening, atol=2e-7, rtol=0.)
    assert abs(results[-1][1]['accounted_j']) < 1e-6
    assert results[-1][1]['dissipated_j'] > 0


def test_unstable_or_invalid_models_and_commands_are_rejected():
    with pytest.raises(ValueError, match='Stable'): three_zone_fixture(spring=100.)
    with pytest.raises(ValueError): SpatialFixture([InterfaceLaw()], [0.], [0., 0.], [1e5, 1e5])
    model = three_zone_fixture(); state = model.virgin()
    with pytest.raises(ValueError): model.segment(state, [0.], 1e-5)
    with pytest.raises(ValueError): model.segment(state, [0., 0.], 0.)
    state.maximum_opening[0] = -1.
    with pytest.raises(ValueError): model.advance(state, [0., 0.])


def test_matched_uniform_histories_are_created_by_remote_drives_not_assignment():
    states = []
    for energy in (1.3, 2.8):
        model = three_zone_fixture(energy)
        law = replace(model.laws[0], area=.01)
        target_damage = .9
        h = law.failure_opening*law.initiation_opening/(law.failure_opening-target_damage*(law.failure_opening-law.initiation_opening))
        remote = h+law.area*law.traction(h, h)/10000.
        state = model.virgin(); model.cycle(state, [remote, remote], 4e-6)
        np.testing.assert_allclose(model.damages(state.maximum_opening), target_damage, atol=1e-13)
        q, maximum, _ = model.equilibrium([.001, .0004], state.maximum_opening)
        assert np.all(maximum == state.maximum_opening)
        states.append(q)
    np.testing.assert_allclose(states[0], states[1], atol=1e-15)
