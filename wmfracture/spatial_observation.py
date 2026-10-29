"""Finite physical-catalog inference and public-observation-only baselines.

The catalog is a discrete prior, NOT a quadrature for a continuous E/history
prior. Every history is prepared through remote loading and unloading. Batch
mechanics below accelerates the fixed three-zone fixture; it is audited against
spatial_fixture, not used as a continuous-domain safety certificate.
"""
from itertools import product
import math

import numpy as np
from scipy.special import logsumexp

from wmfracture.response_quadrature import log_normal_mass


class BatchSpatial:
    """Fixed geometry/standard law, vectorized across parameter hypotheses."""
    def __init__(self, parameters, increments=64):
        self.parameters = np.array(parameters, float, copy=True)
        if (self.parameters.ndim != 2 or self.parameters.shape[1] != 3
                or not len(self.parameters) or not np.isfinite(self.parameters).all()
                or (self.parameters[:, 0] <= .55).any() or (self.parameters[:, 1:] < 0).any()
                or type(increments) is not int or increments < 1):
            raise ValueError('Stable E, two nonnegative preparation drives and positive increment count required')
        self.increments = increments
        self.f = 2*self.parameters[:, 0]/1000.
        self.area = .01/3
        self.modes = np.array(list(product(range(4), repeat=3)), int)
        self.soft = -self.area*1000/(self.f-.0001)
        self.calls = 0

    def damage(self, maximum):
        h = np.maximum(np.asarray(maximum, float), .0001)
        return np.clip(self.f[:, None]*(h-.0001)/(h*(self.f[:, None]-.0001)), 0., 1.)

    def endpoint(self, drive, maximum):
        n = len(self.parameters)
        drive = np.broadcast_to(np.asarray(drive, float), (n, 2))
        maximum = np.asarray(maximum, float)
        if maximum.shape != (n, 3) or not np.isfinite(drive).all() or not np.isfinite(maximum).all() or (maximum < 0).any():
            raise ValueError('Complete finite batch drives and opening histories required')
        old_d = self.damage(maximum)
        slopes = np.stack((np.full((n, 3), self.area*1e8), self.area*1e7*(1-old_d),
            np.broadcast_to(self.soft[:, None], (n, 3)), np.zeros((n, 3))), axis=2)
        offsets = np.zeros_like(slopes); offsets[:, :, 2] = -self.soft[:, None]*self.f[:, None]
        switch = np.maximum(maximum, .0001)
        lower = np.stack((np.full((n, 3), -np.inf), np.zeros((n, 3)), switch,
            np.broadcast_to(self.f[:, None], (n, 3))), axis=2)
        upper = np.stack((np.zeros((n, 3)), switch, np.broadcast_to(self.f[:, None], (n, 3)),
            np.full((n, 3), np.inf)), axis=2)
        k = [slopes[:, i, self.modes[:, i]] for i in range(3)]
        b = [offsets[:, i, self.modes[:, i]] for i in range(3)]
        a00 = 10000.+k[0]+k[1]+k[2]; a11 = 10000.+k[0]+k[2]; a01 = -k[0]+k[2]
        det = a00*a11-a01*a01
        r0 = 5000.*drive.sum(axis=1)[:, None]-b[0]-b[1]-b[2]
        r1 = 5000.*(drive[:, 1]-drive[:, 0])[:, None]+b[0]-b[2]
        q0 = (a11*r0-a01*r1)/det; q1 = (a00*r1-a01*r0)/det
        gaps = (q0-q1, q0, q0+q1)
        valid = np.ones_like(q0, dtype=bool)
        for i in range(3):
            valid &= (gaps[i] >= lower[:, i, self.modes[:, i]]-2e-13)
            valid &= (gaps[i] <= upper[:, i, self.modes[:, i]]+2e-13)
        if not valid.any(axis=1).all():
            raise ArithmeticError('Batch fixture has an inconsistent constitutive mode')
        index = valid.argmax(axis=1); take = np.arange(n)
        q = np.column_stack((q0[take, index], q1[take, index]))
        actual = np.column_stack((q[:, 0]-q[:, 1], q[:, 0], q[:, 0]+q[:, 1]))
        updated = np.maximum(maximum, actual)
        forces = np.where(actual < 0, self.area*1e8*actual, self.area*1e7*(1-self.damage(updated))*actual)
        reaction = 5000.*(drive-np.column_stack((actual[:, 0], actual[:, 2])))
        residual0 = reaction.sum(axis=1)-forces.sum(axis=1)
        residual1 = -reaction[:, 0]+reaction[:, 1]+forces[:, 0]-forces[:, 2]
        if max(np.max(abs(residual0)), np.max(abs(residual1))) > 1e-8:
            raise ArithmeticError('Batch force/moment residual too large')
        self.calls += n
        return q, updated

    def cycle(self, maximum, drive):
        maximum = np.array(maximum, float, copy=True)
        target = np.broadcast_to(np.asarray(drive, float), (len(self.parameters), 2))
        if np.all(target == 0.):
            return np.zeros((len(self.parameters), 2)), maximum
        for fraction in np.arange(1, self.increments+1)/self.increments:
            q, maximum = self.endpoint(target*fraction, maximum)
        peak_q = q.copy()
        for fraction in np.arange(self.increments-1, -1, -1)/self.increments:
            q, maximum = self.endpoint(target*fraction, maximum)
        return peak_q, maximum


class PhysicalCatalog:
    """Public hypotheses and commands only; no true case index or seed."""
    def __init__(self, parameters, actions, futures, increments=64):
        self.physics = BatchSpatial(parameters, increments)
        self.actions = np.array(actions, float, copy=True)
        self.futures = np.array(futures, float, copy=True)
        if (self.actions.ndim != 2 or self.actions.shape[1] != 2
                or self.futures.ndim != 2 or self.futures.shape[1] != 2
                or not np.isfinite(self.actions).all() or not np.isfinite(self.futures).all()
                or (self.actions < 0).any() or (self.futures < 0).any()):
            raise ValueError('Nonnegative finite action and future commands required')
        _, prepared = self.physics.cycle(np.zeros((len(parameters), 3)), self.physics.parameters[:, 1:])
        self.reference_damage = self.physics.damage(prepared)
        self.states = {(): prepared}; self.peaks = {}; self.future_cache = {}

    def transition(self, prefix):
        prefix = tuple(prefix)
        if not prefix: return self.states[()]
        if prefix not in self.states:
            action = prefix[-1]
            if type(action) not in (int, np.int64) or not 0 <= action < len(self.actions):
                raise ValueError('Catalog action index required')
            previous = self.transition(prefix[:-1])
            q, maximum = self.physics.cycle(previous, self.actions[action])
            self.states[prefix] = maximum; self.peaks[prefix] = q
        return self.states[prefix]

    def observations(self, prefix, positions):
        self.transition(prefix)
        q = self.peaks[tuple(prefix)]
        return q[:, 0, None]+q[:, 1, None]*np.asarray(positions)[None, :]/.04

    def damage_increment(self, prefix):
        return np.max(self.physics.damage(self.transition(prefix))-self.reference_damage, axis=1)

    def separated(self, prefix):
        return np.any(self.physics.damage(self.transition(prefix)) == 1., axis=1)

    def future(self, prefix):
        prefix = tuple(prefix)
        if prefix not in self.future_cache:
            maximum = self.transition(prefix)
            gaps, events = [], []
            for drive in self.futures:
                q, final = self.physics.cycle(maximum, drive)
                gaps.append(np.column_stack((q[:, 0]-q[:, 1], q[:, 0], q[:, 0]+q[:, 1])))
                events.append(self.physics.damage(final) == 1.)
            self.future_cache[prefix] = (np.stack(gaps, axis=1), np.stack(events, axis=1))
        return self.future_cache[prefix]


def bounded_bias_posterior(prediction, observed, sigma, bias_sigma, noise_multiple=3., bias_bound=.00015):
    """Finite uniform prior; exact shared-bias/truncated-error likelihood.

    One independent persistent bias per readout channel, truncated Gaussian
    prior on [-bias_bound,bias_bound]; iid Gaussian errors truncated at +/-k SD.
    Omitted normalizers are constant over physical hypotheses at fixed prefix.
    """
    p, y = np.asarray(prediction, float), np.asarray(observed, float)
    if (p.ndim != 3 or y.shape != p.shape[1:] or not len(p) or not p.shape[1]
            or not np.isfinite(p).all() or not np.isfinite(y).all()
            or not np.isfinite([sigma, bias_sigma, noise_multiple, bias_bound]).all()
            or min(sigma, bias_sigma, noise_multiple, bias_bound) <= 0):
        raise ValueError('Finite candidate/frame/channel arrays and positive noise/bias scales required')
    n = p.shape[1]; r = y[None, :, :]-p
    lower = np.maximum(-bias_bound, np.max(r-noise_multiple*sigma, axis=1))
    upper = np.minimum(bias_bound, np.min(r+noise_multiple*sigma, axis=1))
    feasible = np.all(lower <= upper, axis=1)
    positive_mass = np.all(lower < upper, axis=1)
    logw = np.full(len(p), -np.inf)
    if positive_mass.any():
        residual = r[positive_mass]; mean = residual.mean(axis=1)
        gaussian = -.5*(np.sum(((residual-mean[:, None, :])/sigma)**2, axis=(1, 2))
            +np.sum(n*mean**2/(sigma*sigma+n*bias_sigma*bias_sigma), axis=1))
        variance = bias_sigma*bias_sigma*sigma*sigma/(sigma*sigma+n*bias_sigma*bias_sigma)
        center = bias_sigma*bias_sigma*residual.sum(axis=1)/(sigma*sigma+n*bias_sigma*bias_sigma)
        mass = log_normal_mass((lower[positive_mass]-center)/np.sqrt(variance), (upper[positive_mass]-center)/np.sqrt(variance))
        logw[positive_mass] = gaussian+mass.sum(axis=1)
    if not positive_mass.any():
        return {'weights': None, 'feasible': feasible, 'bias_lower_m': lower, 'bias_upper_m': upper}
    weights = np.exp(logw-logsumexp(logw))
    return {'weights': weights, 'feasible': feasible, 'bias_lower_m': lower, 'bias_upper_m': upper}


def infer_public(catalog, actions, observed, positions, sigma, settings):
    actions = tuple(actions)
    if len(actions) != len(observed) or not actions:
        raise ValueError('Complete nonempty action/observation prefix required')
    prediction = np.stack([catalog.observations(actions[:i+1], positions) for i in range(len(actions))], axis=1)
    return bounded_bias_posterior(prediction, observed, sigma, settings['bias_sigma_m'],
        settings['noise_multiple'], settings['bias_bound_m'])


def tilt_proposal(actions, observed, positions, sigma, turn, settings):
    """Uses only initial zero and weak-center observations, never hidden D/E."""
    y = np.asarray(observed, float); x = np.asarray(positions, float)
    if tuple(actions[:2]) != (settings['zero_action'], settings['initial_weak_action']) or y.shape != (len(actions), len(x)):
        raise ValueError('Expected initial zero/weak-center public prefix')
    fallback = settings['left_mid_action'] if turn%2 == 0 else settings['right_mid_action']
    if len(x) != 2 or x[1] <= x[0]: return fallback
    signal = float((y[1, 1]-y[0, 1])-(y[1, 0]-y[0, 0]))
    if abs(signal) <= settings['tilt_threshold_sigma']*sigma: return fallback
    return settings['right_mid_action'] if signal > 0 else settings['left_mid_action']


def choose_public(policy, catalog, actions, observed, positions, sigma, turn, settings):
    posterior = infer_public(catalog, actions, observed, positions, sigma, settings)
    if policy == 'rest_repeat': proposal = settings['zero_action']
    elif policy == 'fixed_lr': proposal = settings['left_mid_action'] if turn%2 == 0 else settings['right_mid_action']
    elif policy == 'fixed_rl': proposal = settings['right_mid_action'] if turn%2 == 0 else settings['left_mid_action']
    elif policy in ('tilt_same_side', 'finite_guard_tilt'):
        proposal = tilt_proposal(actions, observed, positions, sigma, turn, settings)
    else: raise ValueError('Unknown spatial baseline')
    if policy != 'finite_guard_tilt':
        return proposal, posterior, {'proposal': proposal, 'checks': [], 'scope': 'Unfiltered observation-only rule'}
    if not posterior['feasible'].any():
        return None, posterior, {'proposal': proposal, 'checks': [], 'scope': 'Empty finite catalog: abstain, do not assert safety'}
    weak = settings['left_weak_action'] if proposal == settings['left_mid_action'] else settings['right_weak_action']
    checks = []
    for action in (proposal, weak, settings['zero_action']):
        path = tuple(actions)+(action,)
        risk = catalog.damage_increment(path)[posterior['feasible']]
        failed = catalog.separated(path)[posterior['feasible']]
        allowed = bool(np.all(risk <= settings['damage_cap']+1e-12) and not failed.any())
        checks.append({'action': action, 'maximum_finite_damage': float(risk.max()), 'allowed': allowed})
        if allowed:
            return action, posterior, {'proposal': proposal, 'checks': checks,
                'scope': 'Full feasible FINITE catalog scan; not continuous-domain certification'}
    return None, posterior, {'proposal': proposal, 'checks': checks, 'scope': 'No finite-catalog action admitted'}
