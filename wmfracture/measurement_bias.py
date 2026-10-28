"""Standard Gaussian shared-offset observation model, analytically marginalized.

y_i = f_i(theta) + b + epsilon_i, b~N(0,tau^2), epsilon_i~iid N(0,sigma^2).
The SAME offset persists throughout one observed sequence. This does not model
scale, drift, non-Gaussian tracking failures or material-model discrepancy.
"""
from dataclasses import dataclass

import numpy as np

from wmfracture.compliant_inference import log_likelihood


def validate(predictions, observed, sigma, tau):
    predictions, observed = np.asarray(predictions, float), np.asarray(observed, float)
    if (predictions.ndim != 2 or not predictions.shape[1] or observed.shape != (predictions.shape[1],)
            or not np.isfinite(predictions).all() or not np.isfinite(observed).all()
            or not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(tau) or tau < 0):
        raise ValueError('Invalid shared-bias observation inputs')
    return predictions, observed


@dataclass(frozen=True)
class GaussianObservation:
    mode: str = 'independent'
    bias_sigma: float = 0.

    def __post_init__(self):
        if self.mode not in ('independent', 'inflated', 'shared') or not np.isfinite(self.bias_sigma) or self.bias_sigma < 0:
            raise ValueError('Invalid Gaussian observation model')

    def __call__(self, predictions, observed, sigma):
        """Log density up to a theta-independent constant.

        Shared covariance is sigma^2 I + tau^2 11^T. Centered residuals avoid
        cancellation in the Sherman-Morrison quadratic form. Constants omitted
        here cancel in normalized theta weights and Metropolis ratios; these
        scores must NOT be used as normalized evidence across different models.
        """
        predictions, observed = validate(predictions, observed, sigma, self.bias_sigma)
        if self.mode == 'independent' or self.bias_sigma == 0:
            return log_likelihood(predictions, observed, sigma)
        if self.mode == 'inflated':
            return log_likelihood(predictions, observed, np.hypot(sigma, self.bias_sigma))
        residual = observed-predictions
        mean = residual.mean(axis=1)
        centered = residual-mean[:, None]
        n = predictions.shape[1]
        return -.5*(np.sum((centered/sigma)**2, axis=1)
            +n*mean**2/(sigma**2+n*self.bias_sigma**2))

    def bias_moments(self, predictions, observed, sigma, weights):
        """Mixture posterior moments of b, conditional on physical hypotheses."""
        predictions, observed = validate(predictions, observed, sigma, self.bias_sigma)
        weights = np.asarray(weights, float)
        if (weights.shape != (len(predictions),) or not np.isfinite(weights).all()
                or (weights < 0).any() or not np.isclose(weights.sum(), 1.)):
            raise ValueError('Normalized physical-hypothesis weights required')
        if self.mode != 'shared':
            raise ValueError('Only the shared model has one persistent bias variable')
        tau2 = self.bias_sigma**2
        n = predictions.shape[1]
        means = tau2*np.sum(observed-predictions, axis=1)/(sigma**2+n*tau2)
        variance = tau2*sigma**2/(sigma**2+n*tau2)
        mean = float(weights@means)
        total_variance = float(variance+weights@((means-mean)**2))
        return {'mean_m':mean, 'std_m':float(np.sqrt(total_variance)),
            'conditional_std_m':float(np.sqrt(variance)),
            'scope':'Posterior moments, not a Gaussian credible interval for the generally non-Gaussian mixture.'}
