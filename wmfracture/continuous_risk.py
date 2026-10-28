"""Exact-rational box verification for the stable quasistatic scalar fixture.

Conditional on known real-valued binary-float constants, rectangular energy /
remote-preparation bounds, nonnegative fully unloaded cycles and bounded
persistent sensor bias/errors. This is not a certificate for inertial dynamics,
unknown laws, Gaussian tails or real-world safety. No sampled maxima are bounds.
"""
from fractions import Fraction as Q
from copy import deepcopy
import heapq
import math


def rational(value):
    if isinstance(value,Q):
        return value
    if isinstance(value,str):
        return Q(value)
    if not math.isfinite(value):
        raise ValueError('Finite rational input required')
    return Q.from_float(float(value))


def outward(value,upper):
    result = float(value)
    exact = Q.from_float(result)
    if (upper and exact < value) or (not upper and exact > value):
        result = math.nextafter(result,math.inf if upper else -math.inf)
    return result


def box_json(box):
    return [[str(lo),str(hi)] for lo,hi in box]


class ContinuousRisk:
    def __init__(self,inputs,reuse_observations=False):
        if inputs.get('model_kind','stable_quasistatic_nonnegative_cycles')!='stable_quasistatic_nonnegative_cycles':
            raise ValueError('Unsupported model kind; no inertial/dynamic certificate')
        self.inputs = deepcopy(inputs)
        self.box = tuple(tuple(rational(x) for x in pair) for pair in inputs['parameter_bounds'])
        if len(self.box)!=2 or any(len(pair)!=2 or pair[0]>=pair[1] for pair in self.box) or min(pair[0] for pair in self.box)<0:
            raise ValueError('Ordered positive energy/preparation domain required')
        law = inputs['law']
        self.area,self.stiffness,self.traction,self.spring = [rational(v) for v in
            (law['area'],law['tensile_stiffness'],law['peak_traction'],inputs['spring'])]
        if min(self.area,self.stiffness,self.traction,self.spring)<=0:
            raise ValueError('Positive known mechanical constants required')
        self.onset = self.traction/self.stiffness
        self.force_length = self.area*self.traction/self.spring
        self.drive_onset = self.onset+self.force_length
        if 2*self.box[0][0]/self.traction <= self.drive_onset:
            raise ValueError('Whole parameter domain must have a stable unique branch')
        self.actions = tuple(rational(x) for x in inputs['actions_m'])
        self.observed = tuple(rational(x) for x in inputs['observed_gaps_m'])
        count = inputs['reference_count']
        if not self.actions or len(self.actions)!=len(self.observed) or min(self.actions)<0 or type(count) is not int or not 1<=count<=len(self.actions):
            raise ValueError('Nonnegative complete cycles and a fixed reference prefix required')
        self.reference_peak = max(self.actions[:count])
        self.current_peak = max(self.actions)
        running = Q(0);peaks = []
        for action in self.actions:
            running = max(running,action);peaks.append(running)
        self.peaks = tuple(peaks)
        self.radius = rational(inputs['noise_radius_m'])
        self.bias_bound = rational(inputs['bias_bound_m'])
        self.cap = rational(inputs['damage_cap'])
        if min(self.radius,self.bias_bound,self.cap)<0 or self.radius==0:
            raise ValueError('Positive error radius and nonnegative bias/cap required')
        self._damage_cache,self._point_cache = {},{}
        self.point_evaluations = 0
        self.reuse_observations=bool(reuse_observations)
        self._observation_cache={}
        self._point_counts={}
        self.reuse_stats={'box_constraints':0,'point_constraints':0,'box_hits':0,'point_hits':0,
            'box_extensions':0,'point_extensions':0,'context_appends':0}

    def append_observations(self,inputs):
        """Append the ACTUALLY executed action/observation prefix, never a plan.

        Only immutable mechanics/domain/reference/error assumptions may match.
        Cached excluded regions stay excluded; surviving intervals are lazily
        intersected with new observations. No old safe/risk label is reused.
        """
        static=lambda d:{k:v for k,v in d.items() if k not in ('actions_m','observed_gaps_m')}
        count=len(self.actions)
        if (static(inputs)!=static(self.inputs) or len(inputs['actions_m'])<=count
                or inputs['actions_m'][:count]!=self.inputs['actions_m']
                or inputs['observed_gaps_m'][:count]!=self.inputs['observed_gaps_m']):
            raise ValueError('Reuse requires unchanged assumptions and an appended actual prefix')
        checked=ContinuousRisk(inputs,reuse_observations=self.reuse_observations)
        self.inputs=checked.inputs
        self.actions,self.observed,self.peaks=checked.actions,checked.observed,checked.peaks
        self.current_peak=checked.current_peak
        assert self.reference_peak==checked.reference_peak
        if not self.reuse_observations:
            self._point_cache.clear();self._point_counts.clear();self._observation_cache.clear()
        self.reuse_stats['context_appends']+=1

    def damage(self,energy,remote_peak):
        key = (energy,remote_peak)
        if key in self._damage_cache:
            return self._damage_cache[key]
        final = 2*energy/self.traction
        if remote_peak<=self.drive_onset:
            result = Q(0)
        elif remote_peak>=final:
            result = Q(1)
        else:
            result = final*(remote_peak-self.drive_onset)/(final*(remote_peak-self.force_length)-remote_peak*self.onset)
        assert 0<=result<=1
        self._damage_cache[key] = result
        return result

    def gap(self,energy,preparation,action,peak):
        d = self.damage(energy,max(preparation,peak))
        return action*self.spring/(self.spring+self.area*self.stiffness*(1-d))

    def point_support(self,energy,preparation):
        key = (energy,preparation)
        if key in self._point_cache:
            cached=self._point_cache[key]
            old_count=self._point_counts.get(key,len(self.actions))
            if not self.reuse_observations or old_count==len(self.actions) or not cached[0]:
                self.reuse_stats['point_hits']+=1
                return cached
            lower,upper=cached[1:];start=old_count
            self.reuse_stats['point_extensions']+=1
        else:
            lower,upper=-self.bias_bound,self.bias_bound;start=0
        self.point_evaluations += 1
        for action,peak,y in zip(self.actions[start:],self.peaks[start:],self.observed[start:]):
            self.reuse_stats['point_constraints']+=1
            prediction = self.gap(energy,preparation,action,peak)
            lower = max(lower,y-prediction-self.radius)
            upper = min(upper,y-prediction+self.radius)
            if lower>upper:
                break
        result = (lower<=upper,lower,upper)
        self._point_cache[key] = result
        self._point_counts[key]=len(self.actions)
        return result

    def observation_interval(self,box):
        if self.reuse_observations and box in self._observation_cache:
            count,lower,upper=self._observation_cache[box]
            if count==len(self.actions) or lower>upper:
                self.reuse_stats['box_hits']+=1
                return lower,upper
            self.reuse_stats['box_extensions']+=1
        else:
            count,lower,upper=0,-self.bias_bound,self.bias_bound
        (elo,ehi),(hlo,hhi)=box
        for a,peak,y in zip(self.actions[count:],self.peaks[count:],self.observed[count:]):
            self.reuse_stats['box_constraints']+=1
            gap_low=self.gap(ehi,hlo,a,peak);gap_high=self.gap(elo,hhi,a,peak)
            assert gap_low<=gap_high
            lower=max(lower,y-gap_high-self.radius);upper=min(upper,y-gap_low+self.radius)
            if lower>upper:break
        if self.reuse_observations:
            self._observation_cache[box]=(len(self.actions),lower,upper)
        return lower,upper

    def point_risk(self,energy,preparation,action):
        before = self.damage(energy,max(preparation,self.reference_peak))
        after = self.damage(energy,max(preparation,self.current_peak,action))
        return after-before, after==1 and before<1

    def assess(self,box,action):
        """Outer feasibility + upper risk, all computations exact rationals.

        For stable branches D increases with remote peak and decreases with E.
        Each hold gap therefore increases with preparation H and decreases with
        E, giving endpoint observation bounds across all mode boundaries.
        Cumulative D difference is nonincreasing in H: it is constant below the
        reference peak, then decreases, and is zero above the final peak. Use
        H_low for BOTH paths, not independent H intervals. Remaining E dependency
        is bounded by max after-D minus min reference-D and shrinks on splitting.
        """
        (elo,ehi),(hlo,hhi) = box
        lower,upper=self.observation_interval(box)
        if lower>upper:return True,Q(0),False
        before_peak = max(hlo,self.reference_peak)
        after_peak = max(hlo,self.current_peak,action)
        risk = Q(0) if before_peak==after_peak else self.damage(elo,after_peak)-self.damage(ehi,before_peak)
        final_low,final_high = 2*elo/self.traction,2*ehi/self.traction
        final_drive = max(self.current_peak,action)
        fail_upper = min(final_high,final_drive)
        possible_failure = final_low<=fail_upper and fail_upper>max(hlo,self.reference_peak)
        assert 0<=risk<=1
        return False,risk,possible_failure

    def verified_witness(self,point,action):
        energy,preparation = (rational(x) for x in point)
        if not all(lo<=value<=hi for value,(lo,hi) in zip((energy,preparation),self.box)):
            return None
        feasible,lower,upper = self.point_support(energy,preparation)
        if not feasible:
            return None
        risk,failure = self.point_risk(energy,preparation,action)
        return {'parameters':[str(energy),str(preparation)],'bias':str((lower+upper)/2),
            'damage_exact':str(risk),'new_failure':failure}

    def verify(self,action,max_cells=4095,warm_points=()):
        initial_points = self.point_evaluations
        initial_stats=dict(self.reuse_stats)
        action = rational(action)
        if action<0 or type(max_cells) is not int or max_cells<1:
            raise ValueError('Nonnegative query and positive cell budget required')
        nodes,heap = [],[]
        def add(box):
            infeasible,bound,failure = self.assess(box,action)
            index = len(nodes)
            nodes.append({'id':index,'box':box_json(box),'kind':'infeasible' if infeasible else 'open',
                'upper_exact':str(bound),'failure_possible':failure})
            if not infeasible:
                heapq.heappush(heap,(-bound,index,box))
            return index
        add(self.box)
        witness = None
        def consider(point):
            nonlocal witness
            candidate = self.verified_witness(point,action)
            if candidate and (witness is None or candidate['new_failure'] or rational(candidate['damage_exact'])>rational(witness['damage_exact'])):
                witness = candidate
            return candidate and (candidate['new_failure'] or rational(candidate['damage_exact'])>self.cap)
        rejected = any(consider(point) for point in warm_points)
        while heap and not rejected:
            _,index,box = heapq.heappop(heap)
            node = nodes[index]
            centers = tuple((lo+hi)/2 for lo,hi in box)
            samples = [centers]+[(e,h) for e in box[0] for h in box[1]]
            rejected = any(consider(point) for point in samples)
            if rejected:
                break
            if rational(node['upper_exact'])<=self.cap and not node['failure_possible']:
                node['kind'] = 'safe'
                continue
            if len(nodes)+2>max_cells:
                break
            widths = [(hi-lo)/(root_hi-root_lo) for (lo,hi),(root_lo,root_hi) in zip(box,self.box)]
            dimension = 0 if widths[0]>=widths[1] else 1
            middle = (box[dimension][0]+box[dimension][1])/2
            left,right = list(box),list(box)
            left[dimension] = (box[dimension][0],middle)
            right[dimension] = (middle,box[dimension][1])
            children = [add(tuple(left)),add(tuple(right))]
            node.update(kind='split',dimension=dimension,children=children)
        leaves = [n for n in nodes if n['kind']!='split']
        upper = max((rational(n['upper_exact']) for n in leaves),default=Q(0))
        opened = sum(n['kind']=='open' for n in leaves)
        safe = sum(n['kind']=='safe' for n in leaves)
        if rejected:
            decision = 'reject'
        elif opened:
            decision = 'abstain'
        elif witness is not None:
            decision = 'allow'
        elif safe:
            decision = 'abstain'  # risk closed but nonempty support not established
        else:
            decision = 'inconsistent'
        lower = rational(witness['damage_exact']) if witness else None
        return {'decision':decision,'inputs':self.inputs,'action_exact':str(action),'witness':witness,
            'lower_damage':outward(lower,False) if lower is not None else None,'upper_damage':outward(upper,True),
            'upper_damage_exact':str(upper),'evaluated_cells':len(nodes),'open_leaves':opened,'safe_leaves':safe,
            'point_evaluations':self.point_evaluations-initial_points,'proof_nodes':nodes,
            'reuse_stats':{k:self.reuse_stats[k]-v for k,v in initial_stats.items()},
            'scope':'Conditional exact-rational coverage for the supported stable quasistatic scalar model and bounded observation errors. No probabilistic or real-world safety claim.'}


def replay_proof(result,expected_inputs=None,expected_action=None,verifier=None,statistics=None):
    """Recompute proof leaves and exact partition coverage; no claimed maxima trusted."""
    if expected_inputs is not None and result['inputs']!=expected_inputs:
        raise ValueError('Certificate belongs to a different observation context')
    if expected_action is not None and rational(result['action_exact'])!=rational(expected_action):
        raise ValueError('Certificate belongs to a different action')
    if verifier is None:
        verifier=ContinuousRisk(result['inputs'])
    elif type(verifier) is not ContinuousRisk or verifier.inputs!=result['inputs']:
        raise ValueError('Replay cache belongs to a different context or implementation')
    initial_stats=dict(verifier.reuse_stats)
    action = rational(result['action_exact'])
    if action<0:
        raise ValueError('Negative action outside supported model')
    nodes = {n['id']:n for n in result['proof_nodes']}
    if len(nodes)!=len(result['proof_nodes']):
        raise ValueError('Duplicate proof nodes')
    visited = set();bounds = [];open_count = 0;safe_count = 0
    def walk(index,expected):
        nonlocal open_count,safe_count
        if index in visited or index not in nodes:
            raise ValueError('Missing, repeated or cyclic proof node')
        visited.add(index);node = nodes[index]
        box = tuple(tuple(rational(x) for x in pair) for pair in node['box'])
        if box!=expected:
            raise ValueError('Proof partition has a gap or altered boundary')
        if node['kind']=='split':
            dim = node['dimension']
            if dim not in (0,1) or len(node['children'])!=2:
                raise ValueError('Invalid bisection')
            middle = sum(box[dim])/2
            left,right = list(box),list(box)
            left[dim] = (box[dim][0],middle);right[dim] = (middle,box[dim][1])
            walk(node['children'][0],tuple(left));walk(node['children'][1],tuple(right))
            return
        infeasible,upper,failure = verifier.assess(box,action)
        if node['kind']=='infeasible':
            if not infeasible:
                raise ValueError('Unproved feasibility pruning')
        elif node['kind']=='safe':
            if infeasible or upper>verifier.cap or failure:
                raise ValueError('Unproved safe cell')
            safe_count += 1
        elif node['kind']=='open':
            open_count += 1
        else:
            raise ValueError('Unknown proof leaf')
        bounds.append(upper)
    walk(0,verifier.box)
    if len(visited)!=len(nodes):
        raise ValueError('Orphan proof nodes')
    upper = max(bounds,default=Q(0))
    if upper!=rational(result['upper_damage_exact']):
        raise ValueError('Incorrect global risk bound')
    if result['upper_damage']!=outward(upper,True):
        raise ValueError('Incorrect outward-rounded display bound')
    witness = result['witness']
    if witness:
        checked = verifier.verified_witness(witness['parameters'],action)
        if checked is None or checked['damage_exact']!=witness['damage_exact'] or checked['new_failure']!=witness['new_failure']:
            raise ValueError('Invalid lower-bound witness')
        # Verify the submitted bias too, rather than merely existence of some bias.
        energy,preparation = map(rational,witness['parameters'])
        _,lo,hi = verifier.point_support(energy,preparation)
        if not lo<=rational(witness['bias'])<=hi:
            raise ValueError('Witness bias outside observation constraints')
        if result['lower_damage']!=outward(rational(witness['damage_exact']),False):
            raise ValueError('Incorrect lower display bound')
        if rational(witness['damage_exact'])>upper:
            raise ValueError('Lower witness exceeds claimed upper bound')
    elif result['lower_damage'] is not None:
        raise ValueError('Lower bound needs a feasible witness')
    decision = result['decision']
    if decision=='allow' and (open_count or not witness or witness['new_failure'] or upper>verifier.cap):
        raise ValueError('Allow requires complete safe cover and feasible support')
    if decision=='reject' and (not witness or not (witness['new_failure'] or rational(witness['damage_exact'])>verifier.cap)):
        raise ValueError('Reject requires an actual feasible violation')
    if decision=='inconsistent' and (witness or open_count or safe_count):
        raise ValueError('Inconsistency requires complete exclusion')
    if decision not in ('allow','reject','abstain','inconsistent'):
        raise ValueError('Unknown decision')
    if statistics is not None:
        statistics.update({k:verifier.reuse_stats[k]-v for k,v in initial_stats.items()})
    return True
