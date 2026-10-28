"""Quasistatic stable-branch benchmark and standard particle-filter baselines.

This is an analytical reduction of compliant_fixture, not a new damage law.
Only positive, slowly applied remote-displacement cycles with complete unloading
are supported. History is produced by the preparation command, never assigned.
"""
import numpy as np

from wmfracture.cohesive import InterfaceLaw
from wmfracture.compliant_fixture import Fixture


def damage(maximum, energy, law):
    maximum, energy = np.asarray(maximum), np.asarray(energy)
    final = 2*energy/law.peak_traction
    safe = np.maximum(maximum, law.initiation_opening)
    return np.clip(final*(safe-law.initiation_opening)/(safe*(final-law.initiation_opening)), 0., 1.)


def equilibrium_cycle(maximum, energy, drive, law, fixture):
    """Return hold gap and post-unload maximum opening for a monotone cycle.

    For a sufficiently stiff series spring the total displacement envelope is
    strictly increasing, including the cohesive softening branch. Consequently
    the root is unique; unloading preserves its maximum and ends at gap=0.
    """
    maximum, energy, drive = np.broadcast_arrays(np.asarray(maximum, float),
        np.asarray(energy, float), np.asarray(drive, float))
    final, onset = 2*energy/law.peak_traction, law.initiation_opening
    if (not np.isfinite([maximum, energy, drive]).all() or (maximum < 0).any()
            or (drive < 0).any() or (final <= onset).any()):
        raise ValueError('Invalid positive quasistatic cycle')
    soft_slope = law.area*law.peak_traction/(final-onset)
    if (fixture.spring <= soft_slope).any():
        raise ValueError('Nonunique/unstable quasistatic branch is unsupported')
    stiffness = (1-damage(maximum, energy, law))*law.area*law.tensile_stiffness
    unloading = fixture.spring*drive/(fixture.spring+stiffness)
    elastic = fixture.spring*drive/(fixture.spring+law.area*law.tensile_stiffness)
    softening = (fixture.spring*drive-soft_slope*final)/(fixture.spring-soft_slope)
    envelope = np.where(elastic <= onset, elastic, np.where(drive >= final, drive, softening))
    gap = np.where(unloading <= maximum, unloading, envelope)
    return gap, np.maximum(maximum, gap)


class PrefixModel:
    """Inference-facing forward model: parameters + known actions, no labels."""
    def __init__(self, law, fixture, bounds):
        self.law, self.fixture = law, fixture
        self.bounds = np.asarray(bounds, float)
        if self.bounds.shape != (2, 2) or not np.isfinite(self.bounds).all() or (np.diff(self.bounds) <= 0).any():
            raise ValueError('Expected ordered energy / preparation-command bounds')
        self.reset_cost()

    def reset_cost(self):
        self.cost = {'parameter_prefix_evaluations': 0, 'equilibrium_cycle_evaluations': 0}

    def parameters(self, unit):
        unit = np.asarray(unit, float)
        if unit.ndim != 2 or unit.shape[1] != 2 or not np.isfinite(unit).all() or (unit < 0).any() or (unit > 1).any():
            raise ValueError('Parameters must be N x 2 normalized coordinates')
        return self.bounds[:, 0]+unit*np.ptp(self.bounds, axis=1)

    def predict(self, unit, actions):
        physical = self.parameters(unit)
        energy, preparation = physical.T
        _, maximum = equilibrium_cycle(np.zeros(len(unit)), energy, preparation, self.law, self.fixture)
        observations = []
        for action in actions:
            gap, maximum = equilibrium_cycle(maximum, energy, action, self.law, self.fixture)
            observations.append(gap)
        self.cost['parameter_prefix_evaluations'] += len(unit)
        self.cost['equilibrium_cycle_evaluations'] += len(unit)*(1+len(actions))
        return np.stack(observations, axis=1), maximum


def grid(side):
    a, b = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side), indexing='ij')
    return np.column_stack((a.ravel(), b.ravel()))


def log_likelihood(predictions, observed, sigma):
    predictions, observed = np.asarray(predictions), np.asarray(observed)
    if sigma <= 0 or not np.isfinite(sigma) or predictions.ndim != 2 or observed.shape != (predictions.shape[1],):
        raise ValueError('Invalid independent Gaussian observation model')
    return -.5*np.sum(((predictions-observed)/sigma)**2, axis=1)


def normalize(log_weights):
    log_weights = np.asarray(log_weights, float)
    if log_weights.ndim != 1 or not len(log_weights) or not np.isfinite(log_weights).all():
        raise ValueError('Finite nonempty log weights required')
    weights = np.exp(log_weights-np.max(log_weights))
    return weights/weights.sum()


def reflect(unit):
    return 1-np.abs(np.mod(unit, 2)-1)


def systematic_resample(weights, rng):
    weights = np.asarray(weights, float)
    if weights.ndim != 1 or not len(weights) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Normalized weights required')
    cdf = np.cumsum(weights); cdf[-1] = 1.
    return np.searchsorted(cdf, (rng.random()+np.arange(len(weights)))/len(weights))


class ParticleFilter:
    """Static-parameter SMC with complete prefix replay for every MH proposal.

    Mixture of reflected Gaussian random walk and independent uniform proposal
    is symmetric on the uniform-prior unit square. A proposal's history is
    reconstructed from virgin state and ALL observed actions, never by moving
    parameters while retaining the old parameter's material state.
    """
    def __init__(self, model, count, seed, move_steps=2, move_scale=.08, global_probability=.1, likelihood=None):
        self.model, self.rng = model, np.random.default_rng(seed)
        self.unit = self.rng.random((count, 2))
        self.weights = np.ones(count)/count
        self.move_steps, self.move_scale = move_steps, move_scale
        self.global_probability = global_probability
        self.maximum = None
        self.predictions = None
        self.likelihood = log_likelihood if likelihood is None else likelihood
        self.previous_actions = []
        self.previous_observed = []
        self.previous_sigma = None

    def update(self, actions, observed, sigma):
        actions, observed = list(actions), np.asarray(observed, float)
        previous = len(self.previous_actions)
        if (len(actions) != len(observed) or len(actions) <= previous
                or actions[:previous] != self.previous_actions
                or not np.array_equal(observed[:previous], self.previous_observed)
                or (previous and sigma != self.previous_sigma)):
            raise ValueError('SMC updates must append observations to the unchanged prefix/noise model')
        predictions, maximum = self.model.predict(self.unit, actions)
        # Previous weights already contain previous observations exactly once.
        if self.likelihood is log_likelihood:
            incremental = log_likelihood(predictions[:, previous:], observed[previous:], sigma)
        else:
            # Correlated observations do not factor into independent last frames.
            incremental = self.likelihood(predictions, observed, sigma)
            if previous:
                incremental -= self.likelihood(predictions[:, :previous], observed[:previous], sigma)
        self.weights = normalize(np.log(np.maximum(self.weights, np.finfo(float).tiny))+incremental)
        ess = 1/np.sum(self.weights**2)
        resampled = ess < len(self.unit)/2
        if resampled:
            ancestors = systematic_resample(self.weights, self.rng)
            self.unit, maximum = self.unit[ancestors].copy(), maximum[ancestors].copy()
            predictions = predictions[ancestors]
            self.weights[:] = 1/len(self.unit)
        else:
            ancestors = np.arange(len(self.unit))
        target = self.likelihood(predictions, observed, sigma)
        accepted = 0
        for _ in range(self.move_steps):
            proposal = reflect(self.unit+self.rng.normal(0, self.move_scale, self.unit.shape))
            global_move = self.rng.random(len(self.unit)) < self.global_probability
            proposal[global_move] = self.rng.random((global_move.sum(), 2))
            predicted, proposed_maximum = self.model.predict(proposal, actions)
            proposed_target = self.likelihood(predicted, observed, sigma)
            take = np.log(self.rng.random(len(self.unit))) < proposed_target-target
            self.unit[take], maximum[take], target[take] = proposal[take], proposed_maximum[take], proposed_target[take]
            predictions[take] = predicted[take]
            accepted += int(take.sum())
        self.maximum = maximum
        self.predictions = predictions
        self.previous_actions = actions.copy()
        self.previous_observed = observed.copy()
        self.previous_sigma = sigma
        return {'ess_before_resampling':float(ess), 'resampled':bool(resampled),
            'unique_ancestors':len(np.unique(ancestors)), 'accepted_moves':accepted,
            'attempted_moves':self.move_steps*len(self.unit)}


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    cdf = np.cumsum(np.asarray(weights)[order]); cdf[-1] = 1.
    return float(np.asarray(values)[order[np.searchsorted(cdf, quantile)]])
