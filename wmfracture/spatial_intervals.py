"""Outward binary64 interval enclosure of the fixed three-zone model.

Every arithmetic primitive rounds out with nextafter. The model interprets
source float constants as exact reals, with d0=S/K and f=2E/S evaluated as real
ratios. Positive tangent bounds give a unique endpoint root. Branch unions and
history intervals enclose a FIXED finite-increment quasistatic protocol, not an
inertial or continuous-loading fracture trajectory. This is a research checker,
not a formally verified interval library.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np



def down(x): return np.nextafter(np.asarray(x, float), -np.inf)
def up(x): return np.nextafter(np.asarray(x, float), np.inf)


@dataclass
class Interval:
    lo: np.ndarray
    hi: np.ndarray

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(lo if hi is None else hi, float)
        if np.isnan(self.lo).any() or np.isnan(self.hi).any():
            raise ArithmeticError('Indeterminate interval arithmetic; no decision is supported')

    @staticmethod
    def cast(other): return other if isinstance(other, Interval) else Interval(other)

    def __add__(self, other):
        b = self.cast(other)
        return Interval(down(self.lo+b.lo), up(self.hi+b.hi))
    __radd__ = __add__

    def __neg__(self): return Interval(-self.hi, -self.lo)
    def __sub__(self, other): return self+-self.cast(other)
    def __rsub__(self, other): return self.cast(other)+-self

    def __mul__(self, other):
        b = self.cast(other)
        values = np.broadcast_arrays(self.lo*b.lo, self.lo*b.hi, self.hi*b.lo, self.hi*b.hi)
        return Interval(down(np.minimum.reduce(values)), up(np.maximum.reduce(values)))
    __rmul__ = __mul__

    def __truediv__(self, other):
        b = self.cast(other)
        if np.any((b.lo <= 0) & (b.hi >= 0)):
            raise ArithmeticError('Interval division through zero is unsupported')
        return self*Interval(down(1/b.hi), up(1/b.lo))

    def __getitem__(self, index): return Interval(self.lo[index], self.hi[index])


def damage_at_failure(f, h):
    """Monotone-in-h damage at a supplied failure-opening interval."""
    f, h = Interval.cast(f), Interval.cast(h)
    onset = Interval(1000.)/1e7
    safe = Interval(np.maximum(h.lo, onset.lo), np.maximum(h.hi, onset.hi))
    raw = f*(safe-onset)/(safe*(f-onset))
    lo = np.clip(raw.lo, 0., 1.); hi = np.clip(raw.hi, 0., 1.)
    lo = np.where(h.lo >= f.hi, 1., lo); hi = np.where(h.hi <= onset.lo, 0., hi)
    return Interval(lo, hi)


def damage_bounds(energy, history):
    energy = Interval.cast(energy); history = Interval.cast(history)
    f = 2*energy/1000.
    low = damage_at_failure(Interval(f.hi), Interval(history.lo)).lo
    high = damage_at_failure(Interval(f.lo), Interval(history.hi)).hi
    return Interval(low, high)


def damage_difference(energy, reference, after):
    """Coupled max-history difference, avoiding unrelated D extrema.

    For fixed b<=a, D(f,a)-D(f,b) is unimodal in f, maximal at
    f=clip(a,[f_min,f_max]); minima occur at the f endpoints.
    It is nonincreasing in b and nondecreasing in a.
    """
    f = 2*Interval.cast(energy)/1000.
    b, a = Interval.cast(reference), Interval.cast(after)
    af = np.maximum(a.hi, b.lo)
    peak = np.clip(af, f.lo, f.hi)
    upper = (damage_at_failure(Interval(peak), Interval(af))-
             damage_at_failure(Interval(peak), Interval(b.lo))).hi
    lower_parts = []
    for edge in (f.lo, f.hi):
        diff = damage_at_failure(Interval(edge), Interval(a.lo))-damage_at_failure(Interval(edge), Interval(b.hi))
        lower_parts.append(diff.lo)
    lower = np.maximum(0., np.minimum.reduce(lower_parts))
    upper = np.maximum(0., np.minimum(1., upper))
    lower = np.where(a.lo <= b.hi, 0., lower)
    upper = np.where(a.hi <= b.lo, 0., upper)
    return Interval(lower, upper)


class SpatialEnclosure:
    def __init__(self, box, increments=64):
        box = np.asarray(box, float)
        if (box.shape != (3, 2) or not np.isfinite(box).all() or np.any(box[:, 0] > box[:, 1])
                or box[0, 0] <= .55 or np.any(box[1:, 0] < 0)
                or type(increments) is not int or increments < 1):
            raise ValueError('Stable finite E/preparation box and positive increments required')
        self.box = box.copy(); self.energy = Interval(*box[0]); self.increments = increments
        self.f = 2*self.energy/1000.; self.onset = Interval(1000.)/1e7
        self.area = Interval(.01/3); self.soft = -(self.area*1000./(self.f-self.onset))
        self.soft_offset = self.area*1000.*self.f/(self.f-self.onset)
        self.minimum_tangent = float((Interval(10000.)+3*Interval(self.soft.lo)).lo)
        if self.minimum_tangent <= 0: raise ValueError('Positive global tangent bound not established')
        self.modes = np.array(list(product(range(4), repeat=3)), int)

    def endpoint(self, drive, history):
        if np.all(drive.lo == 0.) and np.all(drive.hi == 0.):
            self.last_growth = np.zeros(3, bool)
            return Interval(np.zeros(2)), Interval(history.lo.copy(), history.hi.copy())
        damage = damage_bounds(self.energy, history)
        slopes = [self.area*1e8, self.area*1e7*(1-damage), self.soft, Interval(0.)]
        def branch_value(values, zone):
            los, his = [], []
            for mode in self.modes[:, zone]:
                item = values[mode]
                los.append(float(item.lo if item.lo.ndim == 0 else item.lo[zone]))
                his.append(float(item.hi if item.hi.ndim == 0 else item.hi[zone]))
            return Interval(los, his)
        k = [branch_value(slopes, i) for i in range(3)]
        offsets = [Interval(0.), Interval(0.), self.soft_offset, Interval(0.)]
        b = [branch_value(offsets, i) for i in range(3)]
        a00 = 10000.+k[0]+k[1]+k[2]; a11 = 10000.+k[0]+k[2]; a01 = -k[0]+k[2]
        determinant = a00*a11-a01*a01
        determinant = Interval(np.maximum(determinant.lo, float((Interval(self.minimum_tangent)*self.minimum_tangent).lo)), determinant.hi)
        r0 = 5000.*(drive[0]+drive[1])-b[0]-b[1]-b[2]
        r1 = 5000.*(drive[1]-drive[0])+b[0]-b[2]
        q0 = (a11*r0-a01*r1)/determinant; q1 = (a00*r1-a01*r0)/determinant
        force0 = 5000.*(drive[0]+drive[1]); force1 = 5000.*(drive[1]-drive[0])
        norm = float(((Interval(max(abs(float(force0.lo)), abs(float(force0.hi))))+
                        Interval(max(abs(float(force1.lo)), abs(float(force1.hi)))))/self.minimum_tangent).hi)
        q0 = Interval(np.maximum(q0.lo, -norm), np.minimum(q0.hi, norm))
        q1 = Interval(np.maximum(q1.lo, -norm), np.minimum(q1.hi, norm))
        domain_lo, domain_hi = [], []
        for i in range(3):
            lows = [-np.inf, 0., max(float(history.lo[i]), float(self.onset.lo)), float(self.f.lo)]
            highs = [0., max(float(history.hi[i]), float(self.onset.hi)), float(self.f.hi), np.inf]
            domain_lo.append(np.array(lows)[self.modes[:, i]])
            domain_hi.append(np.array(highs)[self.modes[:, i]])
        # Necessary branch strips, contracted without discarding boundary modes.
        for _ in range(3):
            q0 = Interval(np.maximum(q0.lo, domain_lo[1]), np.minimum(q0.hi, domain_hi[1]))
            p = Interval(domain_lo[0], domain_hi[0])+q1
            q0 = Interval(np.maximum(q0.lo, p.lo), np.minimum(q0.hi, p.hi))
            p = q0-Interval(domain_lo[0], domain_hi[0])
            q1 = Interval(np.maximum(q1.lo, p.lo), np.minimum(q1.hi, p.hi))
            p = Interval(domain_lo[2], domain_hi[2])-q1
            q0 = Interval(np.maximum(q0.lo, p.lo), np.minimum(q0.hi, p.hi))
            p = Interval(domain_lo[2], domain_hi[2])-q0
            q1 = Interval(np.maximum(q1.lo, p.lo), np.minimum(q1.hi, p.hi))
        gaps = [q0-q1, q0, q0+q1]
        valid = (q0.lo <= q0.hi) & (q1.lo <= q1.hi)
        next_lo, next_hi = [], []
        for i, gap in enumerate(gaps):
            lo = np.maximum(gap.lo, domain_lo[i]); hi = np.minimum(gap.hi, domain_hi[i])
            valid &= lo <= hi
            mode = self.modes[:, i]
            hlo = np.maximum(history.lo[i], lo)
            hhi = np.maximum(history.hi[i], hi)
            hlo = np.where(mode == 0, history.lo[i], hlo)
            hhi = np.where(mode == 0, history.hi[i], hhi)
            hhi = np.where(mode == 1, np.maximum(history.hi[i], np.minimum(hi, self.onset.hi)), hhi)
            hlo = np.where(mode == 2, np.maximum(lo, max(history.lo[i], self.onset.lo)), hlo)
            hhi = np.where(mode == 2, np.minimum(hi, self.f.hi), hhi)
            valid &= hlo <= hhi
            next_lo.append(hlo); next_hi.append(hhi)
        if not valid.any(): raise ArithmeticError('All interval constitutive modes excluded')
        self.last_growth = np.array([np.any(valid & (
            ((self.modes[:, i] == 2) & (gaps[i].hi > max(history.lo[i], self.onset.lo))) |
            ((self.modes[:, i] == 3) & (history.lo[i] < self.f.hi)))) for i in range(3)])
        q = Interval([q0.lo[valid].min(), q1.lo[valid].min()], [q0.hi[valid].max(), q1.hi[valid].max()])
        updated = Interval([a[valid].min() for a in next_lo], [a[valid].max() for a in next_hi])
        return q, updated

    def cycle(self, history, drive):
        if np.all(drive.lo == 0.) and np.all(drive.hi == 0.):
            self.cycle_growth = np.zeros(3, bool)
            return Interval(np.zeros(2)), Interval(history.lo.copy(), history.hi.copy())
        growth = np.zeros(3, bool)
        for i in range(1, self.increments+1):
            q, history = self.endpoint(drive*(i/self.increments), history)
            growth |= self.last_growth
        peak = q
        for i in range(self.increments-1, -1, -1):
            _, history = self.endpoint(drive*(i/self.increments), history)
            growth |= self.last_growth
        self.cycle_growth = growth
        return peak, history

    def evaluate(self, actions, positions, proposed):
        prep = Interval(self.box[1:, 0], self.box[1:, 1])
        _, reference = self.cycle(Interval(np.zeros(3)), prep)
        history = Interval(reference.lo.copy(), reference.hi.copy())
        growth = np.zeros(3, bool)
        prediction = []
        for drive in actions:
            pose, history = self.cycle(history, Interval(drive))
            growth |= self.cycle_growth
            channels = [pose[0]+pose[1]*(Interval(x)/.04) for x in positions]
            prediction.append(Interval([float(c.lo) for c in channels], [float(c.hi) for c in channels]))
        _, final = self.cycle(history, Interval(proposed))
        growth |= self.cycle_growth
        risk = damage_difference(self.energy, reference, final)
        risk = Interval(np.where(growth, risk.lo, 0.), np.where(growth, risk.hi, 0.))
        new_separation_possible = bool(np.any(growth & (final.hi >= self.f.lo) & (reference.lo < self.f.hi)))
        return {'prediction': Interval([p.lo for p in prediction], [p.hi for p in prediction]),
            'reference': reference, 'current': history, 'after': final,
            'risk': Interval(float(np.max(risk.lo)), float(np.max(risk.hi))),
            'damage_growth_possible': growth,
            'new_separation_possible': new_separation_possible}


def bias_intervals(prediction, observed, radius, bound, inner=False):
    """Outer: necessary for some state. Inner: sufficient for a POINT state."""
    y = Interval(observed); p = prediction
    if inner:
        low = (y-Interval(p.lo)-radius).hi
        high = (y-Interval(p.hi)+radius).lo
    else:
        low = (y-Interval(p.hi)-radius).lo
        high = (y-Interval(p.lo)+radius).hi
    return np.maximum(-bound, np.max(low, axis=0)), np.minimum(bound, np.min(high, axis=0))
