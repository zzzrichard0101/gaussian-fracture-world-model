"""Clocked zero/frontier baselines with the same conditional risk proof filter.

The schedule is fixed. Probe amplitude uses public observations for feasibility
only; no posterior integration, hypothetical observations or task scores occur.
"""
from wmfracture.task_design import PublicRiskBank,risk_inputs
from wmfracture.verified_policy import RiskReuseSession,filter_proposal


class AlternatingFrontier:
    def __init__(self,zero_first,law,fixture,guard,settings,sigma):
        if type(zero_first) is not bool:raise ValueError('Explicit schedule phase required')
        self.zero_first=zero_first;self.law=law;self.fixture=fixture
        self.guard,self.settings,self.sigma=guard,settings,sigma
        self.session=RiskReuseSession()

    def plan(self,prefix):
        actions=prefix['actions_m'];observed=prefix['observed_gaps_m'];turn=len(actions)-2
        if turn<0 or len(actions)!=len(observed) or actions[:2]!=[0.,self.guard['weak_drive_m']]:
            raise ValueError('Continue the actual initial zero/weak public history')
        zero_phase=(turn%2==0)==self.zero_first
        inputs=risk_inputs(prefix,self.sigma,self.guard,self.law,self.fixture)
        bank=PublicRiskBank(inputs,self.guard,self.law,self.fixture)
        if zero_phase:proposal=0.;valid=None;backtracking=0
        else:
            limit,valid=bank.limit(self.settings['maximum_drive_m'])
            proposal=limit if valid else self.settings['maximum_drive_m']
            backtracking=self.settings['backtracking_steps']
        selected=filter_proposal(inputs,proposal,bank.warm,max_cells=self.settings['cell_budget'],
            backtracking_steps=backtracking,reuse_session=self.session)
        record={'phase':'zero' if zero_phase else 'probe','proposal_m':proposal,'finite_hint_valid':valid,
            'attempts':selected['attempts'],'acquisition_abstained':False,
            'uses_belief_or_task_score':False,'belief_seconds':0.,'score_seconds':0.}
        return selected['action_m'],selected['proof'],record
