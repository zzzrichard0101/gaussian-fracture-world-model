import numpy as np
import pytest

from wmfracture.representation_pilot import Beam, interpolation, gaussian_layout, encoder_matrix, episodes


def test_beam_clamps_and_unforced_energy_decay():
    beam = Beam(17); rng = np.random.default_rng(4); n = len(beam.free)
    state = rng.normal(0,.0001,2*n)
    energy = lambda s: .5*(s[:n]@beam.stiffness@s[:n]+s[n:]@s[n:])
    before = energy(state)
    for _ in range(12): state = beam.step(state,np.zeros(n))
    assert energy(state) < before
    assert np.all(beam.full(state)[[0,1,-2,-1]] == 0.)


def test_action_total_force_does_not_depend_on_visual_representation():
    beam = Beam(17)
    assert beam.force(.45,.02,.07).sum()*beam.h == pytest.approx(.02)


def test_interpolation_reproduces_linear_fields_and_partition_unity():
    x = np.linspace(0,1,15); q = np.linspace(.02,.98,23); matrix = interpolation(x,q)
    np.testing.assert_allclose(matrix.sum(axis=1),1.)
    np.testing.assert_allclose(matrix@(2*x+3),2*q+3)
    with pytest.raises(ValueError): interpolation([0,0],[.2])


def test_all_encoders_preserve_constants_and_are_finite():
    x = np.linspace(0,1,33)
    layout = gaussian_layout([128,'left',1.,5])
    for name in ('point','point_aug','density_aug','weight_aug','kernel_aug','kernel_consistency','grid_aug'):
        matrix = encoder_matrix(layout,x,x,name)
        assert np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix@np.ones(33),1.)


def test_proxy_weights_are_positive_but_not_physical_masses():
    for kind in ('uniform','left','right','jitter'):
        layout = gaussian_layout([64,kind,1.,7])
        assert np.all(layout['opacity']>0) and np.all(layout['area_proxy']>0)
        assert layout['area_proxy'].sum() == pytest.approx(.03)


def test_episode_generation_is_reproducible_and_isolated_by_seed():
    config = {'frames':8,'action_hold_frames':4,'maximum_total_force':.02,'force_width':.07}
    beam = Beam(9)
    a = episodes(beam,2,8,[.2,.8],config)
    b = episodes(beam,2,8,[.2,.8],config)
    c = episodes(beam,2,9,[.2,.8],config)
    np.testing.assert_array_equal(a['states'],b['states'])
    assert not np.array_equal(a['states'],c['states'])
