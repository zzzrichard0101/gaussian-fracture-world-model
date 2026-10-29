"""Replayable box-cover / point-witness decisions for continuous parameters.

Conditional on the fixed finite-increment real quasistatic model enclosed by
spatial_intervals. A sampled point can establish rejection, never permission.
Permission needs an entire safe/pruned cover AND a validated nonempty point.
"""
from copy import deepcopy
import heapq

import numpy as np

from wmfracture.spatial_intervals import SpatialEnclosure, bias_intervals


def validate_inputs(inputs):
    box = np.asarray(inputs['parameter_bounds'], float)
    actions = np.asarray(inputs['actions_m'], float)
    observed = np.asarray(inputs['observed_m'], float)
    positions = np.asarray(inputs['positions_m'], float)
    proposed = np.asarray(inputs['proposed_m'], float)
    if (box.shape != (3, 2) or np.any(box[:, 0] >= box[:, 1])
            or actions.ndim != 2 or actions.shape[1] != 2 or not len(actions)
            or observed.shape != (len(actions), len(positions)) or not len(positions) or proposed.shape != (2,)
            or not all(np.isfinite(a).all() for a in (box, actions, observed, positions, proposed))
            or not np.isfinite([inputs['noise_radius_m'], inputs['bias_bound_m'], inputs['damage_cap']]).all()
            or (actions < 0).any() or (proposed < 0).any()
            or inputs['noise_radius_m'] <= 0 or inputs['bias_bound_m'] <= 0
            or not 0 <= inputs['damage_cap'] < 1):
        raise ValueError('Complete finite public spatial query and positive-volume domain required')
    SpatialEnclosure(box, inputs['increments'])
    return box


def assess(inputs, box):
    model = SpatialEnclosure(box, inputs['increments'])
    result = model.evaluate(inputs['actions_m'], inputs['positions_m'], inputs['proposed_m'])
    lower, upper = bias_intervals(result['prediction'], inputs['observed_m'], inputs['noise_radius_m'], inputs['bias_bound_m'])
    excluded = bool(np.any(lower > upper))
    safe = bool(float(result['risk'].hi) <= inputs['damage_cap'] and not result['new_separation_possible'])
    return {'status': 'excluded' if excluded else 'safe' if safe else 'open',
        'risk_lower': float(result['risk'].lo), 'risk_upper': float(result['risk'].hi),
        'possible_new_separation': result['new_separation_possible'],
        'bias_outer_lower_m': lower.tolist(), 'bias_outer_upper_m': upper.tolist(),
        'prediction_lower_m': result['prediction'].lo.tolist(), 'prediction_upper_m': result['prediction'].hi.tolist()}


def point_evidence(inputs, parameters):
    parameters = np.asarray(parameters, float); bounds = np.asarray(inputs['parameter_bounds'])
    if parameters.shape != (3,) or not np.isfinite(parameters).all() or np.any(parameters < bounds[:, 0]) or np.any(parameters > bounds[:, 1]):
        raise ValueError('Witness must lie inside the declared parameter domain')
    result = SpatialEnclosure(np.column_stack((parameters, parameters)), inputs['increments']).evaluate(
        inputs['actions_m'], inputs['positions_m'], inputs['proposed_m'])
    low, high = bias_intervals(result['prediction'], inputs['observed_m'], inputs['noise_radius_m'], inputs['bias_bound_m'], inner=True)
    feasible = bool(np.all(low <= high))
    return {'parameters': parameters.tolist(), 'feasible': feasible,
        'risk_lower': float(result['risk'].lo), 'risk_upper': float(result['risk'].hi),
        'violating': bool(feasible and float(result['risk'].lo) > inputs['damage_cap']),
        'bias_inner_lower_m': low.tolist(), 'bias_inner_upper_m': high.tolist(),
        'prediction_lower_m': result['prediction'].lo.tolist(), 'prediction_upper_m': result['prediction'].hi.tolist()}


def verify(inputs, max_cells=7, warm_points=()):
    bounds = validate_inputs(inputs)
    if type(max_cells) is not int or max_cells < 1: raise ValueError('Positive cell budget required')
    points = []; feasible_point = False
    for parameters in warm_points:
        evidence = point_evidence(inputs, parameters); points.append(evidence)
        feasible_point |= evidence['feasible']
        if evidence['violating']:
            return {'inputs': deepcopy(inputs), 'decision': 'reject', 'points': points, 'nodes': [],
                'max_cells': max_cells, 'scope': 'Validated point lower bound; fixed-increment conditional model only.'}
    nodes = []; queue = []; root_width = np.ptp(bounds, axis=1)
    def add(box):
        index = len(nodes); row = assess(inputs, box)
        row.update({'box': box.tolist(), 'children': None}); nodes.append(row)
        if row['status'] == 'open':
            volume = float(np.prod(np.ptp(box, axis=1)/root_width))
            heapq.heappush(queue, (-row['risk_upper'], -volume, index))
        return index
    add(bounds)
    while queue and len(nodes)+2 <= max_cells:
        _, _, index = heapq.heappop(queue); node = nodes[index]; box = np.array(node['box'])
        axis = int(np.argmax(np.ptp(box, axis=1)/root_width))
        mid = float(box[axis, 0]+(box[axis, 1]-box[axis, 0])/2)
        if not box[axis, 0] < mid < box[axis, 1]: break
        left, right = box.copy(), box.copy(); left[axis, 1] = mid; right[axis, 0] = mid
        node['children'] = [add(left), add(right)]; node['split_axis'] = axis; node['split_value'] = mid
    leaves = [n for n in nodes if n['children'] is None]
    if all(n['status'] == 'excluded' for n in leaves): decision = 'inconsistent'
    elif all(n['status'] in ('safe', 'excluded') for n in leaves) and feasible_point: decision = 'allow'
    else: decision = 'abstain'
    return {'inputs': deepcopy(inputs), 'decision': decision, 'points': points, 'nodes': nodes, 'max_cells': max_cells,
        'scope': 'Full continuous-parameter box cover for a fixed finite-increment model; not continuous-loading, inertial or real-world safety.'}


def replay(proof, expected_inputs=None):
    inputs = proof['inputs']; bounds = validate_inputs(inputs)
    if expected_inputs is not None and inputs != expected_inputs: raise ValueError('Public query mismatch')
    points = [point_evidence(inputs, p['parameters']) for p in proof['points']]
    if points != proof['points']: raise ValueError('Invalid point evidence')
    if proof['decision'] == 'reject':
        if not any(p['violating'] for p in points): raise ValueError('Rejection has no validated feasible violation')
        return True
    nodes = proof['nodes']
    if not nodes or nodes[0]['box'] != bounds.tolist(): raise ValueError('Missing root cover')
    reached = set(); leaves = []
    def visit(index, expected_box):
        if index in reached or not 0 <= index < len(nodes): raise ValueError('Invalid proof tree')
        reached.add(index); node = nodes[index]
        if node['box'] != expected_box.tolist(): raise ValueError('Partition gap or wrong child box')
        recomputed = assess(inputs, expected_box)
        if any(node[k] != v for k, v in recomputed.items()): raise ValueError('Forged enclosure claim')
        if node['children'] is None: leaves.append(node); return
        axis, split = node['split_axis'], node['split_value']
        if len(node['children']) != 2 or not expected_box[axis, 0] < split < expected_box[axis, 1]: raise ValueError('Invalid split')
        left, right = expected_box.copy(), expected_box.copy(); left[axis, 1] = split; right[axis, 0] = split
        visit(node['children'][0], left); visit(node['children'][1], right)
    visit(0, bounds)
    if len(reached) != len(nodes): raise ValueError('Unreachable proof nodes')
    if all(n['status'] == 'excluded' for n in leaves): decision = 'inconsistent'
    elif all(n['status'] in ('safe', 'excluded') for n in leaves) and any(p['feasible'] for p in points): decision = 'allow'
    else: decision = 'abstain'
    if proof['decision'] != decision: raise ValueError('Decision does not follow from evidence')
    return True
