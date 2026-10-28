"""Standard proposal/backtracking filter with replayed continuous-region proofs.

The proposal and witness callback receive public information only. Unknown
queries may reduce the search interval but are never called unsafe witnesses.
Only an allow certificate permits an action. This does not optimize task utility.
"""
import time

from wmfracture.continuous_risk import ContinuousRisk,replay_proof


class RiskReuseSession:
    """Separate trusted search/checker caches; neither trusts submitted bounds."""
    def __init__(self):
        self.search=None;self.checker=None

    def context(self,inputs):
        if self.search is None:
            self.search=ContinuousRisk(inputs,reuse_observations=True)
            self.checker=ContinuousRisk(inputs,reuse_observations=True)
        elif self.search.inputs!=inputs:
            # Validate both before mutating either cache.
            for item in (self.search,self.checker):
                static=lambda d:{k:v for k,v in d.items() if k not in ('actions_m','observed_gaps_m')}
                n=len(item.actions)
                if (static(inputs)!=static(item.inputs) or len(inputs['actions_m'])<=n
                        or inputs['actions_m'][:n]!=item.inputs['actions_m']
                        or inputs['observed_gaps_m'][:n]!=item.inputs['observed_gaps_m']):
                    raise ValueError('Session cannot reuse changed assumptions or a changed past')
            self.search.append_observations(inputs);self.checker.append_observations(inputs)
        assert self.search is not self.checker
        return self.search,self.checker


def filter_proposal(inputs,proposal,warm_points,max_cells=1023,backtracking_steps=8,reuse_session=None):
    if proposal<0 or type(backtracking_steps) is not int or backtracking_steps<0:
        raise ValueError('Nonnegative proposal and backtracking budget required')
    if reuse_session is None:
        verifier,checker=ContinuousRisk(inputs),None
    else:
        verifier,checker=reuse_session.context(inputs)
    attempts=[]
    def check(action):
        started=time.perf_counter();warm=warm_points(action);warm_seconds=time.perf_counter()-started
        started=time.perf_counter();result=verifier.verify(action,max_cells=max_cells,warm_points=warm)
        solve_seconds=time.perf_counter()-started
        replay_statistics={}
        started=time.perf_counter();replay_proof(result,expected_inputs=inputs,expected_action=action,
            verifier=checker,statistics=replay_statistics)
        replay_seconds=time.perf_counter()-started
        attempts.append({'action_m':action,'decision':result['decision'],'lower_damage':result['lower_damage'],
            'upper_damage':result['upper_damage'],'cells':result['evaluated_cells'],'points':result['point_evaluations'],
            'witness':result['witness'],'warm_seconds':warm_seconds,'solve_seconds':solve_seconds,
            'reuse_stats':result['reuse_stats'],
            'replay_stats':replay_statistics,
            'proof_replay_seconds':replay_seconds})
        return result
    first=check(proposal)
    if first['decision']=='allow':
        return {'action_m':proposal,'proof':first,'attempts':attempts,'stopped':False}
    best=check(0.)
    if best['decision']!='allow':
        return {'action_m':None,'proof':None,'attempts':attempts,'stopped':True}
    low,high=0.,proposal
    for _ in range(backtracking_steps):
        middle=(low+high)/2;result=check(middle)
        if result['decision']=='allow':
            low,best=middle,result
        else:
            high=middle
    return {'action_m':low,'proof':best,'attempts':attempts,'stopped':False}
