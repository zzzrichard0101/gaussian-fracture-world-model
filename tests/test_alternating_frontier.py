from copy import deepcopy
import json
from pathlib import Path

import pytest

from wmfracture.alternating_frontier import AlternatingFrontier
from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture
from wmfracture.continuous_risk import replay_proof
from wmfracture.task_design import risk_inputs


def settings():
    root=Path(__file__).parents[1]
    s=json.loads((root/'config/task-aware-pilot.json').read_text());s.update(cell_budget=127,backtracking_steps=3)
    g=json.loads((root/'config/finite-damage-guard.json').read_text());g['grid_side']=9
    return s,g


@pytest.mark.parametrize('zero_first',[True,False])
def test_both_phases_keep_reference_and_execute_replayed_actual_input(zero_first):
    s,g=settings();law,fixture=InterfaceLaw(area=.01),Fixture()
    controller=AlternatingFrontier(zero_first,law,fixture,g,s,.00002)
    prefix={'actions_m':[0.,.001],'observed_gaps_m':[0.,.0005]}
    for turn in range(4):
        frozen=deepcopy(prefix);action,proof,record=controller.plan(prefix)
        assert prefix==frozen and action is not None
        assert record['phase']==('zero' if ((turn%2==0)==zero_first) else 'probe')
        if record['phase']=='zero':assert action==0.
        assert replay_proof(proof,expected_inputs=risk_inputs(prefix,.00002,g,law,fixture),expected_action=action)
        assert proof['inputs']['reference_count']==2 and not record['uses_belief_or_task_score']
        # A valid noisy response, enough to continue the same public context.
        prefix['actions_m'].append(action);prefix['observed_gaps_m'].append(action*.5)


def test_schedule_does_not_access_future_or_belief_settings():
    s,g=settings()
    class OnlyRiskSettings(dict):
        def __getitem__(self,key):
            if key not in ('maximum_drive_m','backtracking_steps','cell_budget'):
                raise AssertionError('Simple baseline accessed inference/task settings')
            return super().__getitem__(key)
    controller=AlternatingFrontier(False,InterfaceLaw(area=.01),Fixture(),g,OnlyRiskSettings(s),.00002)
    action,proof,_=controller.plan({'actions_m':[0.,.001],'observed_gaps_m':[0.,.0005]})
    assert action is not None and replay_proof(proof)


def test_rewritten_past_and_invalid_prefix_rejected():
    s,g=settings();controller=AlternatingFrontier(True,InterfaceLaw(area=.01),Fixture(),g,s,.00002)
    prefix={'actions_m':[0.,.001],'observed_gaps_m':[0.,.0005]}
    controller.plan(prefix)
    changed={'actions_m':[0.,.001,0.],'observed_gaps_m':[0.,.00051,0.]}
    with pytest.raises(ValueError):controller.plan(changed)
    with pytest.raises(ValueError):controller.plan({'actions_m':[.1,.001],'observed_gaps_m':[0.,0.]})


def test_rotated_collection_reproduces_prior_future_policy_without_changing_reference(tmp_path,monkeypatch):
    root=Path(__file__).parents[1];monkeypatch.syspath_prepend(str(root/'scripts'))
    import task_aware_pilot as previous
    import alternating_frontier_evaluate as current
    from history_refinement import digest
    s,g=settings();base=json.loads((root/'config/observation-budget.json').read_text());base.update(s);s=base
    s.update(policies=['future_task'],budgets=[2],gap_noise_sigma_m=[.00001,.00002,.00003],
        actual_bias_m=[.00003],noise_conditions=[{'name':'iid_bounded','latent_rho':0.}])
    cases=[{'id':'new-000','parameters':[2.,.0016],'family':'test','cluster':'test'}]
    oldroot,newroot=tmp_path/'old',tmp_path/'new'
    for folder in (oldroot,newroot):(folder/'proofs').mkdir(parents=True)
    law,fixture=InterfaceLaw(area=.01),Fixture()
    previous.collect(oldroot,s,g,cases,law,fixture)
    source=oldroot/'public-episodes.json';before=digest(source);reference=json.loads(source.read_text())
    s['policies']=['future_task','alternating_zero_first','alternating_probe_first']
    result=current.collect(newroot,s,g,cases,law,fixture,reference)
    assert result['reproduced_future_task_checkpoints']==3
    assert all(positions==[1,1,1] for positions in result['position_counts'].values())
    assert digest(source)==before
    public=json.loads((newroot/'public-episodes.json').read_text())
    assert len(public)==9 and all(e['budget_fulfilled'] for e in public)
