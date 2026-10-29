"""Response-conditioned local proposals, with density correction in physical space.

This is NOT an assumed globally invertible reaction-coordinate map. Local
Gaussian charts and a defensive uniform component define a known proposal q
on normalized physical parameters. The unchanged uniform-prior posterior is
estimated with likelihood/q. Plateau/rank-deficient regions are not removed.
Proposal derivatives are numerical heuristics, never interval bounds.
"""
import numpy as np
from scipy.special import logsumexp, ndtri

from wmfracture.response_quadrature import log_normal_mass
from wmfracture.spatial_observation import PhysicalCatalog


def bounded_loglike(prediction, observed, sigma, settings):
    """Same bounded-error/shared-bias target as bounded_bias_posterior."""
    p, y = np.asarray(prediction, float), np.asarray(observed, float)
    if p.ndim != 3 or y.shape != p.shape[1:] or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Finite parameter/frame/channel predictions required')
    tau, multiple, bound = [settings[k] for k in ('bias_sigma_m', 'noise_multiple', 'bias_bound_m')]
    if min(sigma, tau, multiple, bound) <= 0: raise ValueError('Positive noise/bias scales required')
    n = p.shape[1]; residual = y[None]-p
    lower = np.maximum(-bound, np.max(residual-multiple*sigma, axis=1))
    upper = np.minimum(bound, np.min(residual+multiple*sigma, axis=1))
    positive = np.all(lower < upper, axis=1); result = np.full(len(p), -np.inf)
    if positive.any():
        r = residual[positive]; mean = r.mean(axis=1)
        gaussian = -.5*(np.sum(((r-mean[:, None])/sigma)**2, axis=(1, 2))
            +np.sum(n*mean**2/(sigma*sigma+n*tau*tau), axis=1))
        variance = tau*tau*sigma*sigma/(sigma*sigma+n*tau*tau)
        center = tau*tau*r.sum(axis=1)/(sigma*sigma+n*tau*tau)
        mass = log_normal_mass((lower[positive]-center)/np.sqrt(variance), (upper[positive]-center)/np.sqrt(variance))
        result[positive] = gaussian+mass.sum(axis=1)
    return result


class DefensiveMixture:
    def __init__(self, centers, covariances, uniform_fraction=.2):
        self.centers = np.array(centers, float, copy=True)
        self.covariances = np.array(covariances, float, copy=True)
        self.alpha = float(uniform_fraction)
        if (self.centers.ndim != 2 or self.centers.shape[1] != 3 or not len(self.centers)
                or self.covariances.shape != (len(self.centers), 3, 3)
                or not np.isfinite(self.centers).all() or not np.isfinite(self.covariances).all()
                or not 0 < self.alpha < 1
                or not np.allclose(self.covariances, self.covariances.transpose(0, 2, 1), atol=1e-14)):
            raise ValueError('Finite 3D Gaussian charts and a positive defensive component required')
        self.cholesky = np.linalg.cholesky(self.covariances)
        self.inverse = np.linalg.inv(self.cholesky)
        self.logdet = np.log(np.diagonal(self.cholesky, axis1=1, axis2=2)).sum(axis=1)

    def log_density(self, points):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Finite 3D proposal points required')
        delta = points[:, None, :]-self.centers[None]
        whitened = np.einsum('kij,nkj->nki', self.inverse, delta)
        gaussian = -.5*np.sum(whitened**2, axis=2)-self.logdet[None]-1.5*np.log(2*np.pi)
        mixture = np.log(1-self.alpha)-np.log(len(self.centers))+logsumexp(gaussian, axis=1)
        inside = np.all((points >= 0.) & (points <= 1.), axis=1)
        prior = np.where(inside, np.log(self.alpha), -np.inf)
        return np.logaddexp(prior, mixture)

    def sample(self, unit4):
        unit4 = np.asarray(unit4, float)
        if unit4.ndim != 2 or unit4.shape[1] != 4 or not np.isfinite(unit4).all() or np.any(unit4 < 0.) or np.any(unit4 >= 1.):
            raise ValueError('Four open/half-open unit coordinates required')
        prior = unit4[:, 0] < self.alpha
        points = unit4[:, 1:].copy()
        selected = np.flatnonzero(~prior)
        if len(selected):
            component = np.minimum(((unit4[selected, 0]-self.alpha)/(1-self.alpha)*len(self.centers)).astype(int), len(self.centers)-1)
            # Values on the measure-zero inverse-CDF endpoints are assigned a
            # finite representative; no Gaussian proposal is renormalized.
            normal = ndtri(np.clip(unit4[selected, 1:], np.nextafter(0., 1.), np.nextafter(1., 0.)))
            points[selected] = self.centers[component]+np.einsum('nij,nj->ni', self.cholesky[component], normal)
        return points, self.log_density(points)


def summarize_importance(loglike, log_proposal, gaps, events):
    loglike, log_proposal = np.asarray(loglike), np.asarray(log_proposal)
    logw = loglike-log_proposal; good = np.isfinite(logw)
    result = {'available':bool(good.any()), 'nodes':len(logw), 'positive_weight_nodes':int(good.sum()), 'ess':0.,
        'mean_gap_m':None, 'event_probability':None, 'risk_upper_bound':None}
    if good.any():
        weights = np.exp(logw-logsumexp(logw))
        result.update({'ess':float(1/np.sum(weights*weights)),
            'mean_gap_m':np.einsum('n,nij->ij', weights, gaps).tolist(),
            'event_probability':(weights@np.any(events, axis=2)).tolist()})
    return result


class PublicForward:
    def __init__(self, bounds, actions, futures, positions, increments=64):
        self.bounds = np.asarray(bounds, float); self.actions = actions; self.futures = futures
        self.positions = positions; self.increments = increments
        self.parameter_evaluations = 0; self.endpoint_parameter_evaluations = 0

    def evaluate(self, unit, prefix, future=False):
        unit = np.asarray(unit, float)
        parameters = self.bounds[:, 0]+unit*np.ptp(self.bounds, axis=1)
        bank = PhysicalCatalog(parameters, self.actions, self.futures, self.increments)
        prediction = np.stack([bank.observations(tuple(prefix[:i+1]), self.positions) for i in range(len(prefix))], axis=1)
        output = bank.future(tuple(prefix)) if future else None
        self.parameter_evaluations += len(unit)
        self.endpoint_parameter_evaluations += bank.physics.calls
        return prediction, output


def whiten(values, sigma, tau):
    """L^-1 over time, independently for each observation channel."""
    values = np.asarray(values, float); time = values.shape[1]
    factor = np.linalg.cholesky(sigma*sigma*np.eye(time)+tau*tau*np.ones((time, time)))
    inverse = np.linalg.inv(factor)
    return np.einsum('ij,njc->nic', inverse, values).reshape(len(values), -1)


def numerical_jacobian(forward, centers, prefix, step):
    count = len(centers); probes = []
    for axis in range(3):
        plus, minus = centers.copy(), centers.copy()
        plus[:, axis] += step; minus[:, axis] -= step
        probes.extend((plus, minus))
    prediction, _ = forward.evaluate(np.concatenate(probes), prefix)
    blocks = prediction.reshape(6, count, *prediction.shape[1:])
    return np.stack([(blocks[2*i]-blocks[2*i+1])/(2*step) for i in range(3)], axis=-1)


def fit_proposals(forward, pilot_unit, pilot_prediction, prefix, observed, sigma, settings, config):
    count = config['charts']; target = np.asarray(observed)
    # Proposal adaptation only: a smooth enlarged-noise likelihood. The final
    # posterior always uses the original bounded likelihood and 1/q correction.
    adapt_sigma = max(sigma, config['adaptation_sigma_floor_m'])
    residual = whiten(target[None]-pilot_prediction, adapt_sigma, settings['bias_sigma_m'])
    pilot_log = -.5*np.sum(residual**2, axis=1); weights = np.exp(pilot_log-logsumexp(pilot_log))
    mean = weights@pilot_unit; centered = pilot_unit-mean
    covariance = (centered.T*weights)@centered*config['physical_covariance_scale']**2
    covariance += np.eye(3)*config['physical_std_floor']**2
    # Uniform E anchors cover the otherwise unresolved material direction;
    # no anchor's probability is used as a physical prior mass.
    energies = (np.arange(count)+.5)/count
    initial_response = pilot_prediction[:, 1]-pilot_prediction[:, 0]
    target_response = target[1]-target[0]
    response_score = np.sum(((initial_response-target_response)/adapt_sigma)**2, axis=1)
    centers = []
    for energy in energies:
        index = int(np.argmin(response_score+((pilot_unit[:, 0]-energy)/config['anchor_energy_window'])**2))
        centers.append([energy, *pilot_unit[index, 1:]])
    raw = np.asarray(centers); projected = raw.copy(); audits = []
    for iteration in range(config['projection_steps']):
        active_prefix = prefix[:2] if iteration < config['initial_response_steps'] else prefix
        active_target = target[:len(active_prefix)]
        pred, _ = forward.evaluate(projected, active_prefix)
        jac = numerical_jacobian(forward, projected, active_prefix, config['derivative_step_unit'])
        white_jac = np.stack([whiten(jac[:, :, :, i], sigma, settings['bias_sigma_m']) for i in range(3)], axis=2)
        residual = whiten(active_target[None]-pred, sigma, settings['bias_sigma_m'])
        u, singular, vt = np.linalg.svd(white_jac, full_matrices=False)
        gain = singular/(singular*singular+config['projection_damping']**2)
        delta = np.einsum('nki,nk->ni', vt, gain*np.einsum('nmk,nm->nk', u, residual))
        if len(active_prefix) == 2: delta[:, 0] = 0.
        largest = np.max(abs(delta), axis=1)
        delta *= np.minimum(1., config['maximum_projection_step']/np.maximum(largest, 1e-300))[:, None]
        score = np.sum(residual*residual, axis=1); best = projected.copy(); best_score = score.copy()
        for fraction in config['backtracking_fractions']:
            proposed = np.clip(projected+fraction*delta, config['center_margin'], 1-config['center_margin'])
            fitted, _ = forward.evaluate(proposed, active_prefix)
            candidate_score = np.sum(whiten(active_target[None]-fitted, sigma, settings['bias_sigma_m'])**2, axis=1)
            mask = candidate_score < best_score
            best[mask] = proposed[mask]; best_score[mask] = candidate_score[mask]
        projected = best
        audits.append({'iteration':iteration, 'frames':len(active_prefix), 'median_whitened_squared_residual':float(np.median(best_score))})
    jac = numerical_jacobian(forward, projected, prefix, config['derivative_step_unit'])
    white = np.stack([whiten(jac[:, :, :, i], sigma, settings['bias_sigma_m']) for i in range(3)], axis=2)
    _, singular, vt = np.linalg.svd(white, full_matrices=False)
    std = np.clip(config['response_scale']/np.maximum(singular, 1e-300), config['response_std_floor'], config['null_std'])
    chart_covariance = np.einsum('nki,nk,nkj->nij', vt, std*std, vt)
    rank_threshold = np.maximum(config['rank_absolute_tolerance'], singular[:, :1]*config['rank_relative_tolerance'])
    ranks = np.sum(singular > rank_threshold, axis=1)
    proposals = {
        'physical_mixture':DefensiveMixture(raw, np.broadcast_to(covariance, (count,3,3)), config['uniform_fraction']),
        'projected_physical':DefensiveMixture(projected, np.broadcast_to(covariance, (count,3,3)), config['uniform_fraction']),
        'response_chart':DefensiveMixture(projected, chart_covariance, config['uniform_fraction'])}
    return proposals, {'projection':audits, 'rank_counts':{str(i):int(np.sum(ranks == i)) for i in range(4)},
        'raw_centers_unit':raw.tolist(), 'projected_centers_unit':projected.tolist(),
        'singular_values':singular.tolist(), 'chart_standard_deviations_unit':std.tolist()}
