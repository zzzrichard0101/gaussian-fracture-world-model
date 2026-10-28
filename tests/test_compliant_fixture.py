from dataclasses import asdict
import math

import numpy as np
import pytest

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture,DriveState,advance,segment,cycle,energy,clone_state,nominal_remote_peak


def test_remote_elastic_equilibrium_and_work_balance():
    law,fixture,state = InterfaceLaw(area=.01),Fixture(),DriveState()
    _,audit = segment(state,law,fixture,.00005,.2,2e-5)
    segment(state,law,fixture,.00005,.1,2e-5)
    expected = fixture.spring/(fixture.spring+law.area*law.tensile_stiffness)*.00005
    assert state.gap == pytest.approx(expected,abs=1e-10)
    assert law.damage(state.maximum_opening) == 0.
    assert abs(energy(state,law,fixture)+state.damping_heat-state.actuator_work) < 1e-9
    assert audit['maximum_energy_residual_j'] < 1e-9


def test_same_fixture_preparation_does_not_assign_nominal_damage():
    law = InterfaceLaw(area=.01)
    stable,soft = Fixture(),Fixture(spring=1000.)
    states = []
    for fixture in (stable,soft):
        state = DriveState()
        cycle(state,law,fixture,nominal_remote_peak(law,fixture,.001),4e-5,ramp=.2,hold=.05,rest=.1)
        states.append(state)
    assert abs(states[0].maximum_opening-.001) < 1e-5
    assert states[1].maximum_opening < law.initiation_opening
    assert law.damage(states[1].maximum_opening) == 0.
    assert states[0].time > .5


def test_restart_clones_every_state_field_and_matches_continuous_steps():
    law,fixture = InterfaceLaw(area=.01),Fixture()
    state = DriveState()
    for _ in range(1000):
        advance(state,law,fixture,.00003,2e-5)
    restored = clone_state(state)
    assert asdict(restored) == asdict(state) and restored is not state
    for _ in range(1000):
        advance(state,law,fixture,.00005,2e-5)
        advance(restored,law,fixture,.00005,2e-5)
    assert asdict(restored) == asdict(state)


def test_unload_preserves_damage_and_clock_with_no_hidden_reset():
    law,fixture,state = InterfaceLaw(area=.01),Fixture(),DriveState()
    cycle(state,law,fixture,.0018,4e-5,ramp=.2,hold=.05,rest=.1)
    before = clone_state(state)
    cycle(state,law,fixture,.001,4e-5,ramp=.2,hold=.05,rest=.1)
    assert state.maximum_opening >= before.maximum_opening
    assert law.damage(state.maximum_opening) >= law.damage(before.maximum_opening)
    assert state.time > before.time and state.step > before.step and state.damping_heat >= before.damping_heat
    assert abs(state.gap) < 1e-8 and abs(state.velocity) < 1e-7


def test_invalid_clocks_or_unstable_steps_fail():
    law,fixture = InterfaceLaw(area=.01),Fixture()
    with pytest.raises(ValueError):
        advance(DriveState(),law,fixture,.001,.01)
    with pytest.raises(ValueError):
        segment(DriveState(),law,fixture,.001,.100001,2e-5)
    with pytest.raises(ValueError):
        nominal_remote_peak(law,fixture,law.failure_opening)
