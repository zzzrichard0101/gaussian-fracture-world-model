"""QMC ratio integration under a uniform CONTINUOUS physical prior.

Equal-volume Sobol nodes reuse the batch physics cache. Their normalized
likelihood weights approximate integrals; unlike the previous 29-state model,
adding nodes does not change the prior. These samples are never risk upper
bounds or a conservative feasible-set representation.
"""
import numpy as np

from wmfracture.spatial_observation import bounded_bias_posterior


def estimate(prediction, gaps, events, observed, sigma, settings):
    posterior = bounded_bias_posterior(prediction, observed, sigma, settings['bias_sigma_m'],
        settings['noise_multiple'], settings['bias_bound_m'])
    weights = posterior['weights']
    result = {'nodes': len(prediction), 'feasible_nodes': int(posterior['feasible'].sum()),
        'available': weights is not None, 'ess': 0., 'mean_gap_m': None, 'event_probability': None,
        'sample_envelope_lower_m': None, 'sample_envelope_upper_m': None,
        'risk_upper_bound': None,
        'scope': 'Uniform-continuous-prior QMC ratio estimate. Sample extrema are not outer bounds or calibrated intervals.'}
    if weights is None:
        result['status'] = 'no_feasible_nodes_not_domain_inconsistent'
        return result
    result.update({'status': 'sample_estimate_available_not_convergence_proven',
        'ess': float(1/np.sum(weights*weights)),
        'mean_gap_m': np.einsum('n,nij->ij', weights, gaps).tolist(),
        'event_probability': (weights@np.any(events, axis=2)).tolist(),
        'sample_envelope_lower_m': gaps[posterior['feasible']].min(axis=0).tolist(),
        'sample_envelope_upper_m': gaps[posterior['feasible']].max(axis=0).tolist()})
    return result
