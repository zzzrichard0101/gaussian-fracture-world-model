"""Explicit physics-budget accounting and defensive broad/response proposals."""
import numpy as np
from scipy.special import logsumexp

from wmfracture.spatial_transport import DefensiveMixture, whiten


def path_endpoint_cost(actions, futures, prefix, increments, include_future=True):
    """Scalar parameter/end-step solves; positive preparation assumed."""
    cycles = 1 + sum(np.any(np.asarray(actions[a]) != 0) for a in prefix)
    if include_future:
        cycles += sum(np.any(np.asarray(f) != 0) for f in futures)
    return int(2*increments*cycles)


def main_allowance(budget, setup, unit_cost):
    if (type(budget) is not int or type(setup) is not int or type(unit_cost) is not int
            or min(budget, unit_cost) <= 0 or setup < 0):
        raise ValueError('Integer positive budget/unit cost and nonnegative setup required')
    if setup > budget:
        raise ValueError('Setup exceeds declared budget; cannot spend future budget')
    return (budget-setup)//unit_cost


def physical_pilot_proposal(unit, prediction, observed, sigma, settings, config):
    """The old raw physical_mixture, without unnecessary projection physics."""
    adapt_sigma = max(sigma, config['adaptation_sigma_floor_m'])
    residual = whiten(np.asarray(observed)[None]-prediction, adapt_sigma, settings['bias_sigma_m'])
    loglike = -.5*np.sum(residual**2, axis=1)
    weights = np.exp(loglike-logsumexp(loglike))
    mean = weights@unit; centered = unit-mean
    covariance = (centered.T*weights)@centered*config['physical_covariance_scale']**2
    covariance += np.eye(3)*config['physical_std_floor']**2
    response = prediction[:, 1]-prediction[:, 0]
    target = np.asarray(observed)[1]-np.asarray(observed)[0]
    score = np.sum(((response-target)/adapt_sigma)**2, axis=1)
    centers = []
    for energy in (np.arange(config['charts'])+.5)/config['charts']:
        index = int(np.argmin(score+((unit[:, 0]-energy)/config['anchor_energy_window'])**2))
        centers.append([energy, *unit[index, 1:]])
    return DefensiveMixture(centers, np.broadcast_to(covariance, (config['charts'], 3, 3)), config['uniform_fraction'])


def equal_hybrid(broad, response):
    """Equal conditional Gaussian mass; one unchanged uniform component."""
    if broad.alpha != response.alpha or len(broad.centers) != len(response.centers):
        raise ValueError('Same defensive fraction and equal chart counts required')
    return DefensiveMixture(np.concatenate((broad.centers, response.centers)),
        np.concatenate((broad.covariances, response.covariances)), broad.alpha)


def select_physics_draws(proposal, unit4, count, chunk=1024):
    """Whole-mixture cube conditioning; stopping never uses likelihood."""
    if type(count) is not int or count <= 0 or type(chunk) is not int or chunk <= 0:
        raise ValueError('Positive integer accepted count and chunk required')
    pieces, densities, masks = [], [], []
    accepted = 0
    for start in range(0, len(unit4), chunk):
        unit = unit4[start:start+chunk]
        if proposal is None:
            points, logq = unit[:, 1:].copy(), np.zeros(len(unit))
        else:
            points, logq = proposal.sample(unit)
        inside = np.all((points >= 0)&(points <= 1), axis=1)
        if accepted+int(inside.sum()) >= count:
            end = int(np.flatnonzero(inside)[count-accepted-1])+1
            pieces.append(points[:end]); densities.append(logq[:end]); masks.append(inside[:end])
            return np.concatenate(pieces), np.concatenate(densities), np.concatenate(masks)
        pieces.append(points); densities.append(logq); masks.append(inside)
        accepted += int(inside.sum())
    raise ValueError('Predeclared proposal stream exhausted before physics allowance')


def cutoff_for_allowance(inside, count):
    if type(count) is not int or count <= 0 or np.count_nonzero(inside) < count:
        raise ValueError('Positive feasible physics allowance required')
    return int(np.flatnonzero(inside)[count-1])+1
