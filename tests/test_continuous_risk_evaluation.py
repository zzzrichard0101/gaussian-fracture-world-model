import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.compliant_inference import equilibrium_cycle
from wmfracture.damage_guard import FiniteGuard
from wmfracture.continuous_risk import replay_proof


def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import continuous_risk_evaluate
    return continuous_risk_evaluate


def test_bounded_sampler_rejects_tails_but_gaussian_control_does_not(monkeypatch):
    module=modules(monkeypatch);law,fixture=InterfaceLaw(area=.01),Fixture();sigma=.00002
    for family,expected in (('bounded_normal',2*sigma),('gaussian',4*sigma)):
        specimen=module.EvaluationSpecimen([1.5,.0016],0.,sigma,1,law,fixture,
            noise_family=family,noise_multiple=3.)
        sequence=iter([4*sigma,2*sigma])
        specimen.rng=SimpleNamespace(normal=lambda *args:next(sequence))
        specimen.observe(0.)
        assert specimen.trace[-1]['noise_m']==expected


def test_new_cases_are_reproducible_off_grid_and_matched_controls_are_valid(monkeypatch):
    module=modules(monkeypatch);project=Path(__file__).parents[1]
    settings=json.loads((project/'config/continuous-risk-evaluation.json').read_text())
    config=json.loads((project/'config/finite-damage-guard.json').read_text());config['grid_side']=settings['grid_side']
    law,fixture=InterfaceLaw(area=.01),Fixture()
    cases=module.new_cases(settings,config,law,fixture,[])
    assert len(cases)==24 and cases==module.new_cases(settings,config,law,fixture,[])
    assert len({c['cluster'] for c in cases})==20
    for cluster in {c['cluster'] for c in cases if c['family']=='matched_pair'}:
        pair=[c for c in cases if c['cluster']==cluster];gaps=[]
        for case in pair:
            energy,prep=case['parameters']
            _,maximum=equilibrium_cycle(0.,energy,prep,law,fixture)
            gap,_=equilibrium_cycle(maximum,energy,.001,law,fixture);gaps.append(gap)
        assert gaps[0]==pytest.approx(float(gaps[1]),abs=1e-14)
    with pytest.raises(AssertionError):
        module.new_cases(settings,config,law,fixture,[cases[0]])


def test_policy_decision_needs_no_private_case_or_future_label(tmp_path,monkeypatch):
    module=modules(monkeypatch);project=Path(__file__).parents[1]
    config=json.loads((project/'config/finite-damage-guard.json').read_text());config['grid_side']=7
    settings={'grid_side':7,'warm_point_count':3,'cell_budget':63,'backtracking_steps':2}
    law,fixture=InterfaceLaw(area=.01),Fixture()
    prefix={'actions_m':[0.,.001],'observed_gaps_m':[0.,.0005]};sigma=.00002
    result=module.decide(tmp_path,'anonymous',0,'verified_filter',FiniteGuard(law,fixture,config,sigma),
        prefix,sigma,settings,config,law,fixture)
    assert not (tmp_path/'private-traces.json').exists()
    if result['action_m'] is not None:
        proof=json.loads((tmp_path/result['executed_proof']['file']).read_text())
        assert proof['decision']=='allow'
        assert replay_proof(proof,expected_action=result['action_m'])
    else:
        assert result['executed_proof'] is None and result['filter']['stopped']
