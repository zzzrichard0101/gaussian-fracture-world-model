import json
from pathlib import Path

import numpy as np
import pytest

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.guided_quadrature import GuidedResponseQuadrature


def test_guided_partition_preserves_prior_mass_and_energy_event():
    model=GuidedResponseQuadrature(InterfaceLaw(area=.01),Fixture(),[[1.2,3.],[.0009,.0023]],
        [0.],[0.],.000001,.00005)
    result=model.integrate([.0031],energy_panels=4,inner_order=8)
    assert np.exp(result.log_integral)==pytest.approx(1.,abs=1e-12)
    assert result.weights@result.parameters[:,0]==pytest.approx(2.1,abs=1e-12)
    assert model.forecast(result,[.0031])[0]['separation_mass']==pytest.approx((1.55-1.2)/1.8,abs=1e-12)


@pytest.mark.parametrize('index',[0,1])
def test_public_narrow_energy_ridge_matches_adaptive_outer_reference(index):
    data=json.loads((Path(__file__).parent/'fixtures/response-quadrature-ridges.json').read_text())
    case=data['cases'][index];episode=case['episode'];guard=data['guard']
    model=GuidedResponseQuadrature(InterfaceLaw(**data['law']),Fixture(**data['fixture']),guard['parameter_bounds'],
        episode['actions_m'],episode['observed_gaps_m'],episode['sigma_m'],guard['bias_sigma_m'])
    commands=[f['drive_m'] for f in case['reference']]
    fine=model.integrate(commands,energy_panels=16,energy_order=8,inner_order=32)
    prediction=model.forecast(fine,commands)
    np.testing.assert_allclose([f['mean_gap_m'] for f in prediction],[f['mean_gap_m'] for f in case['reference']],atol=2e-8)
    np.testing.assert_allclose([f['separation_mass'] for f in prediction],[f['separation_mass'] for f in case['reference']],atol=2e-5)
    cuts=model.energy_cuts(commands,8)
    assert np.all(np.diff(cuts)>0) and cuts[0]==1.2 and cuts[-1]==3.
