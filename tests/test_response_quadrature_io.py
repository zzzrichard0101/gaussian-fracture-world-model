import json
from pathlib import Path

from wmfracture.cohesive import InterfaceLaw
from dataclasses import asdict


def test_public_forecast_seals_before_regenerated_label_evaluation(tmp_path,monkeypatch):
    root=Path(__file__).parents[1];monkeypatch.syspath_prepend(str(root/'scripts'))
    import response_quadrature_evaluate as module
    settings=json.loads((root/'config/response-quadrature.json').read_text())
    settings['levels']=[{'name':'response_32','energy_panels':2,'energy_order':4,'inner_order':4},
        {'name':'response_64','energy_panels':4,'energy_order':4,'inner_order':8}]
    settings['adaptive_audit_case_indices']=[]
    guard=json.loads((root/'config/finite-damage-guard.json').read_text())
    old=json.loads((root/'config/observation-budget.json').read_text())
    bundle={'guard':guard,'source_settings':old,'law':asdict(InterfaceLaw(area=.01)),
        'case_settings':json.loads((root/'config/continuous-risk-evaluation.json').read_text()),
        'baseline_forecasts':[],
        'episodes':[{'id':'test','context_id':'b-000-0-0-0','trajectory_id':'test',
            'policy':'weak_repeat','sigma_m':.00002,'noise_condition':'iid_bounded','budget':2,
            'actions_m':[0.,.001,.001,.001],'observed_gaps_m':[.00003,.00063,.00063,.00063]}]}
    for name,value in [('config.json',settings),('inputs.json',bundle),('code-sha256.json',{})]:
        module.write_json(tmp_path/name,value)
    labels=bundle['case_settings']
    class ForbiddenLabels:
        def __getitem__(self,key):raise AssertionError('Inference read synthetic label generation settings')
    bundle['case_settings']=ForbiddenLabels()
    seals=module.forecast_public(tmp_path,bundle,settings)
    assert (tmp_path/'sealed-predictions.json').is_file()
    bundle['case_settings']=labels
    groups,rows=module.evaluate(tmp_path,bundle,settings,seals)
    assert len(groups)==len(rows)==2
    assert {r['method'] for r in rows}=={'response_32','response_64'}
