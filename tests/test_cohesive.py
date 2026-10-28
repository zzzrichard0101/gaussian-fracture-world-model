from dataclasses import asdict
import json
import math

import numpy as np
import pytest

from wmfracture.cohesive import InterfaceLaw, make_system, prescribed_cycle, save_state, load_state


def test_monotonic_traction_integrates_to_specified_fracture_energy():
    law = InterfaceLaw()
    gaps = np.linspace(0, law.failure_opening, 40001)
    traction = np.array([law.traction(gap, gap) for gap in gaps])
    assert np.trapz(traction, gaps) == pytest.approx(law.fracture_energy, rel=1e-8)
    assert law.dissipated(law.failure_opening) == pytest.approx(law.fracture_energy * law.area)


def test_loading_unloading_no_healing_and_compression_does_not_damage():
    system = make_system()
    prescribed_cycle(system, .002)
    kappa = system.state.maximum_opening
    damage = system.law.damage(kappa)
    assert 0 < damage < 1
    assert system.opening == 0
    assert system.state.v_left == system.state.v_right == 0
    prescribed_cycle(system, .001)
    assert system.state.maximum_opening == kappa
    system.prescribe_opening(-.0001)
    assert system.law.traction(system.opening, kappa) == pytest.approx(-1e4)
    assert system.state.maximum_opening == kappa
    assert system.law.damage(kappa) == damage
    assert system.law.traction(.001, kappa) == pytest.approx((1-damage) * 1e4)


def test_prescribed_cycle_work_is_dissipation_at_rest_and_zero_gap():
    system = make_system()
    prescribed_cycle(system, .002, samples=10000)
    assert system.state.external_work == pytest.approx(system.energies()["dissipated"], rel=1e-5)


def test_compression_only_oscillator_matches_analytical_nonadhesive_contact():
    system = make_system()
    # Known failed interface is a material-point unit initial condition only.
    system.state.maximum_opening = system.law.failure_opening
    speed = .2
    system.state.v_left, system.state.v_right = .5 * speed, -.5 * speed
    omega = math.sqrt(system.law.area * system.law.compressive_stiffness / system.reduced_mass)
    end = math.pi / omega
    dt = end / 10000
    initial_momentum = system.momentum()
    minimum = 0.
    for _ in range(10000):
        system.advance(dt)
        minimum = min(minimum, system.opening)
    assert minimum == pytest.approx(-speed / omega, rel=1e-6)
    assert system.state.v_right - system.state.v_left == pytest.approx(speed, rel=1e-6)
    assert system.momentum() == pytest.approx(initial_momentum, abs=1e-13)
    assert system.law.damage(system.state.maximum_opening) == 1.
    assert system.law.traction(.001, system.state.maximum_opening) == 0.


def test_initial_relative_energy_predicts_separation_or_bound_motion():
    for velocity, should_break in [(.02, False), (.25, False), (.6, True)]:
        system = make_system()
        system.impulse(left=velocity)
        energy = system.energies()["accounted_total"]
        momentum = system.momentum()
        for _ in range(16000):
            system.advance(5e-6)
        broken = system.law.damage(system.state.maximum_opening) == 1.
        assert broken == should_break
        assert system.momentum() == pytest.approx(momentum, abs=1e-12)
        assert system.energies()["accounted_total"] == pytest.approx(energy, rel=5e-4)
        if broken:
            expected = math.sqrt(velocity ** 2 - 2 * system.law.fracture_energy * system.law.area / system.reduced_mass)
            assert system.state.v_right - system.state.v_left == pytest.approx(expected, rel=5e-4)


def test_exact_restart_of_damaged_state_and_integrity_guard(tmp_path):
    system = make_system()
    prescribed_cycle(system, .002)
    system.impulse(left=.25)
    for _ in range(400):
        system.advance(5e-6)
    path = tmp_path / "state.json"
    save_state(path, system)
    restored = load_state(path)
    for _ in range(400):
        system.advance(5e-6)
        restored.advance(5e-6)
    assert asdict(system) == asdict(restored)
    with pytest.raises(FileExistsError):
        save_state(path, system)
    corrupt = json.loads(path.read_text())
    corrupt["payload"]["state"]["maximum_opening"] = 0.
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="corrupt"):
        load_state(path)


def test_history_pair_reaches_same_visible_state_through_valid_boundary_motion():
    pristine, damaged = make_system(), make_system()
    prescribed_cycle(pristine, 0.)
    prescribed_cycle(damaged, .002)
    assert (pristine.state.x_left, pristine.state.x_right, pristine.state.v_left, pristine.state.v_right) == (
        damaged.state.x_left, damaged.state.x_right, damaged.state.v_left, damaged.state.v_right)
    assert 0 < damaged.law.damage(damaged.state.maximum_opening) < 1
    for system in (pristine, damaged):
        system.impulse(left=.245)
        for _ in range(20000):
            system.advance(5e-6)
    assert pristine.law.damage(pristine.state.maximum_opening) < 1
    assert damaged.law.damage(damaged.state.maximum_opening) == 1


@pytest.mark.parametrize("kwargs", [{"area": -1}, {"fracture_energy": .00001}, {"peak_traction": float("nan")}])
def test_rejects_invalid_law(kwargs):
    with pytest.raises(ValueError):
        InterfaceLaw(**kwargs)
