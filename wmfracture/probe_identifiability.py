"""Exact two-hypothesis amplitude diagnostic, not a full-domain policy bound.

Nonnegative, fully unloaded, stable quasistatic cycles only. E is fracture
energy; H is a REMOTE preparation peak, not a joint opening. Rational branch
endpoints give the exact maximum of the piecewise-affine gap difference.
Gaussian discrimination scores are a separate illustrative noise model.
"""
import math
from fractions import Fraction

import numpy as np
from scipy.special import ndtr

from wmfracture.continuous_risk import rational


class PairProbeDiagnostic:
    def __init__(self, verifier):
        self.model = verifier

    def state(self, parameters):
        if len(parameters) != 2:
            raise ValueError('One energy and remote preparation peak required')
        e, h = map(rational, parameters)
        m = self.model
        if not (m.box[0][0] <= e <= m.box[0][1] and m.box[1][0] <= h <= m.box[1][1]):
            raise ValueError('Diagnostic state must belong to the declared prior')
        if h >= 2*e/m.traction:
            raise ValueError('Already separated preparations are excluded')
        return e, h

    def compliance(self, energy, remote_peak):
        m = self.model
        d = m.damage(energy, remote_peak)
        return m.spring/(m.spring+m.area*m.stiffness*(1-d))

    def matched_preparation(self, energy, compliance):
        """Invert q(E,H) on the softening branch, retaining exact rationals."""
        e, q = rational(energy), rational(compliance)
        m = self.model
        q0 = m.spring/(m.spring+m.area*m.stiffness)
        if not q0 < q < 1:
            raise ValueError('Compliance must lie strictly on the softening branch')
        f = 2*e/m.traction
        alpha = (f-m.onset)/(f-m.drive_onset)
        beta = m.force_length*f/(f-m.drive_onset)
        h = beta/(alpha-q)
        self.state((e, h))
        return h

    def amplitude_ceiling(self, parameters, maximum_drive):
        """Pair-oracle constraint only. Separation is explicitly excluded.

        This diagnostic requires D_ref+cap<1 so the ceiling is a closed safe
        interval, rather than an open endpoint at complete separation.
        """
        e, h = self.state(parameters)
        m = self.model
        limit = rational(maximum_drive)
        if limit < 0:
            raise ValueError('Nonnegative amplitude limit required')
        reference = m.damage(e, max(h, m.reference_peak))
        target = reference+m.cap
        if target >= 1:
            raise ValueError('Closed nonseparating diagnostic requires D_ref+cap<1')
        f = 2*e/m.traction
        # Invert D(E,R)=f(R-U0)/(f(R-c)-R*d0).
        ceiling = f*(m.drive_onset-target*m.force_length)/(f-target*(f-m.onset))
        if max(h, m.current_peak) > ceiling:
            raise ValueError('Existing prefix already exceeds the damage budget')
        return min(limit, ceiling)

    def maximum_difference(self, first, second, maximum_drive):
        m = self.model
        states = [self.state(first), self.state(second)]
        ceiling = min(self.amplitude_ceiling(p, maximum_drive) for p in states)
        cuts = {Fraction(0), ceiling}
        for e, h in states:
            # On either side of each of these points y(a) is affine.
            cuts.update(x for x in (m.drive_onset, max(h, m.current_peak), 2*e/m.traction) if 0 < x < ceiling)
        candidates = []
        for a in sorted(cuts):
            gaps = [m.gap(e, h, a, max(m.current_peak, a)) for e, h in states]
            risks = [m.point_risk(e, h, a) for e, h in states]
            assert all(r <= m.cap and not separated for r, separated in risks)
            candidates.append({'action': a, 'difference': abs(gaps[0]-gaps[1]), 'gaps': gaps,
                               'damage_increments': [r for r, _ in risks]})
        best = max(candidates, key=lambda row: row['difference'])
        return {'ceiling': ceiling, 'best': best, 'candidates': candidates,
                'scope': 'Exact two-state continuous-amplitude maximum. Other feasible states may forbid these actions; not a full-belief safe policy or global inference limit.'}

    def sequence_difference(self, first, second, actions):
        states = [self.state(first), self.state(second)]
        m = self.model
        peak = m.current_peak
        delta = []
        for value in actions:
            a = rational(value)
            if a < 0:
                raise ValueError('Nonnegative complete cycles required')
            peak = max(peak, a)
            if any(m.point_risk(e, h, peak)[0] > m.cap or m.point_risk(e, h, peak)[1] for e, h in states):
                raise ValueError('Sequence violates a pair damage/separation constraint')
            delta.append(float(m.gap(*states[0], a, peak)-m.gap(*states[1], a, peak)))
        return np.asarray(delta)


def gaussian_pair_score(delta, sigma, bias_sigma=0.):
    """Equal-prior Bayes error for two fixed mean vectors, common covariance.

    Covariance = sigma^2 I + bias_sigma^2 11^T; same Gaussian bias prior under
    both hypotheses. No truncated/correlated-noise or minimax-bias guarantee.
    This is oracle binary discrimination, not unknown-state posterior accuracy.
    """
    delta = np.asarray(delta, float)
    if (delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all()
            or not math.isfinite(sigma) or sigma <= 0
            or not math.isfinite(bias_sigma) or bias_sigma < 0):
        raise ValueError('Finite difference vector and positive noise required')
    # Centered decomposition avoids subtracting two nearly equal large terms.
    mean = float(delta.mean())
    d2 = float(np.sum(((delta-mean)/sigma)**2)+len(delta)*mean**2/(sigma**2+len(delta)*bias_sigma**2))
    return {'mahalanobis_squared': d2, 'kl_nats': d2/2,
            'equal_prior_bayes_error': float(ndtr(-math.sqrt(d2)/2))}
