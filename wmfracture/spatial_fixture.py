"""Small-rotation rigid bar on cohesive zones and compliant remote drives.

q=[translation, length_scale*rotation] has two length-valued coordinates.
Each zone retains its own maximum opening. At fixed prior history the local
incremental force is piecewise affine; all mode combinations are enumerated.
A conservative positive-definite tangent bound excludes unstable instances.
This is a quasistatic, displacement-increment model, not a dynamic fracture
solver or an extension of the scalar continuous-risk certificate.
"""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np

from wmfracture.cohesive import InterfaceLaw


@dataclass
class SpatialState:
    q: np.ndarray
    drive: np.ndarray
    maximum_opening: np.ndarray
    step: int = 0
    actuator_work: float = 0.

    def clone(self):
        return SpatialState(self.q.copy(), self.drive.copy(), self.maximum_opening.copy(),
                            self.step, self.actuator_work)


class SpatialFixture:
    def __init__(self, laws, positions, drive_positions, springs, length_scale=.04):
        self.laws = tuple(laws)
        self.positions = np.array(positions, dtype=float, copy=True)
        self.drive_positions = np.array(drive_positions, dtype=float, copy=True)
        self.springs = np.array(springs, dtype=float, copy=True)
        self.length_scale = float(length_scale)
        if (not self.laws or any(not isinstance(law, InterfaceLaw) for law in self.laws)
                or self.positions.shape != (len(self.laws),)
                or self.drive_positions.ndim != 1 or len(self.drive_positions) < 2
                or self.springs.shape != self.drive_positions.shape
                or not all(np.isfinite(a).all() for a in (self.positions, self.drive_positions, self.springs))
                or not math.isfinite(self.length_scale) or self.length_scale <= 0 or (self.springs <= 0).any()):
            raise ValueError('Finite zones and at least two positive compliant drives required')
        self.B = np.column_stack((np.ones(len(self.laws)), self.positions/self.length_scale))
        self.C = np.column_stack((np.ones(len(self.springs)), self.drive_positions/self.length_scale))
        self.drive_stiffness = self.C.T@(self.springs[:, None]*self.C)
        soft = np.array([law.area*law.peak_traction/(law.failure_opening-law.initiation_opening) for law in self.laws])
        self.tangent_lower = self.drive_stiffness-self.B.T@(soft[:, None]*self.B)
        self.minimum_tangent_bound = float(np.linalg.eigvalsh(self.tangent_lower)[0])
        if self.minimum_tangent_bound <= 1e-10*max(1., float(np.linalg.norm(self.drive_stiffness, 2))):
            raise ValueError('Stable unique incremental branch is not established; use stiffer drives')
        self.gap_tolerance = 2e-13

    def virgin(self):
        return SpatialState(np.zeros(2), np.zeros(len(self.springs)), np.zeros(len(self.laws)))

    def validate(self, state):
        if (state.q.shape != (2,) or state.drive.shape != self.springs.shape
                or state.maximum_opening.shape != self.positions.shape
                or not all(np.isfinite(a).all() for a in (state.q, state.drive, state.maximum_opening))
                or (state.maximum_opening < 0).any()
                or (self.B@state.q > state.maximum_opening+self.gap_tolerance).any()
                or type(state.step) is not int or state.step < 0 or not math.isfinite(state.actuator_work)):
            raise ValueError('Invalid spatial state/history')

    def damages(self, maximum):
        return np.array([law.damage(float(h)) for law, h in zip(self.laws, maximum)])

    def zone_forces(self, q, maximum):
        return np.array([law.area*law.traction(float(g), float(h))
                         for law, g, h in zip(self.laws, self.B@q, maximum)])

    def drive_forces(self, state):
        return self.springs*(state.drive-self.C@state.q)

    def energies(self, state):
        self.validate(state)
        elastic = .5*np.sum(self.springs*(state.drive-self.C@state.q)**2)
        elastic += sum(law.potential(float(g), float(h)) for law, g, h in zip(self.laws, self.B@state.q, state.maximum_opening))
        dissipated = sum(law.dissipated(float(h)) for law, h in zip(self.laws, state.maximum_opening))
        return {'elastic_j': float(elastic), 'dissipated_j': float(dissipated),
                'accounted_j': float(elastic+dissipated-state.actuator_work)}

    def _branches(self, law, history):
        # (slope, intercept, lower opening, upper opening, mode label)
        branches = [(law.area*law.compressive_stiffness, 0., -np.inf, 0., 'contact')]
        if history >= law.failure_opening:
            return branches+[(0., 0., 0., np.inf, 'separated')]
        upper = max(history, law.initiation_opening)
        branches.append((law.area*law.tensile_stiffness*(1-law.damage(history)), 0., 0., upper, 'unloading'))
        slope = -law.area*law.peak_traction/(law.failure_opening-law.initiation_opening)
        branches.append((slope, -slope*law.failure_opening, upper, law.failure_opening, 'softening'))
        branches.append((0., 0., law.failure_opening, np.inf, 'separated'))
        return branches

    def equilibrium(self, drive, maximum):
        """Unique endpoint equilibrium for FIXED old maximum openings.

        An increment can cross internal path switches. Segment refinement is
        still required to resolve any intervening maxima and work integration.
        """
        drive, maximum = np.asarray(drive, float), np.asarray(maximum, float)
        if (drive.shape != self.springs.shape or maximum.shape != self.positions.shape
                or not np.isfinite(drive).all() or not np.isfinite(maximum).all() or (maximum < 0).any()):
            raise ValueError('Complete finite drive and old history required')
        combinations = list(product(*(self._branches(law, float(h)) for law, h in zip(self.laws, maximum))))
        modes = np.array([[[v for v in branch[:4]] for branch in combo] for combo in combinations])
        stiffness = self.drive_stiffness+np.einsum('ni,mn,nj->mij', self.B, modes[:, :, 0], self.B)
        rhs = self.C.T@(self.springs*drive)-modes[:, :, 1]@self.B
        solutions = np.linalg.solve(stiffness, rhs[:, :, None])[:, :, 0]
        gaps = solutions@self.B.T
        valid = np.all((gaps >= modes[:, :, 2]-self.gap_tolerance)
                       & (gaps <= modes[:, :, 3]+self.gap_tolerance), axis=1)
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise ArithmeticError('No consistent constitutive mode; do not execute this increment')
        chosen = int(indices[0]); q = solutions[chosen]
        if np.max(abs(solutions[indices]-q)) > 100*self.gap_tolerance:
            raise ArithmeticError('Ambiguous equilibria despite declared stability bound')
        updated = np.maximum(maximum, self.B@q)
        reaction = self.springs*(drive-self.C@q)
        residual = self.C.T@reaction-self.B.T@self.zone_forces(q, updated)
        if np.max(abs(residual)) > 1e-8:
            raise ArithmeticError('Equilibrium residual too large')
        return q, updated, {'force_residual_n': float(residual[0]),
            'moment_residual_nm': float(residual[1]*self.length_scale),
            'minimum_tangent_eigenvalue_n_m': float(np.linalg.eigvalsh(stiffness[chosen])[0]),
            'modes': [branch[4] for branch in combinations[chosen]]}

    def advance(self, state, drive):
        self.validate(state)
        q, maximum, audit = self.equilibrium(drive, state.maximum_opening)
        drive = np.asarray(drive, float)
        before = self.drive_forces(state)
        after = self.springs*(drive-self.C@q)
        state.actuator_work += float(.5*(before+after)@(drive-state.drive))
        state.q = q.copy(); state.drive = drive.copy(); state.maximum_opening = maximum.copy()
        state.step += 1
        self.validate(state)
        return audit

    def snapshot(self, state):
        self.validate(state)
        return {'step': state.step, 'q_m': state.q.tolist(), 'rotation_rad': float(state.q[1]/self.length_scale),
                'drive_m': state.drive.tolist(), 'gaps_m': (self.B@state.q).tolist(),
                'maximum_opening_m': state.maximum_opening.tolist(),
                'damage': self.damages(state.maximum_opening).tolist(),
                'drive_forces_n': self.drive_forces(state).tolist(),
                'actuator_work_j': state.actuator_work, **self.energies(state)}

    def segment(self, state, target, maximum_drive_increment_m=4e-6):
        self.validate(state)
        target = np.asarray(target, float)
        if (target.shape != self.springs.shape or not np.isfinite(target).all()
                or not math.isfinite(maximum_drive_increment_m) or maximum_drive_increment_m <= 0):
            raise ValueError('Finite target and positive displacement resolution required')
        start = state.drive.copy()
        count = max(1, math.ceil(float(np.max(abs(target-start)))/maximum_drive_increment_m))
        initial = self.energies(state)['accounted_j']
        rows = [self.snapshot(state)]
        audits = []
        for i in range(1, count+1):
            audit = self.advance(state, start+(target-start)*(i/count))
            row = self.snapshot(state); row.update(audit); rows.append(row); audits.append(audit)
        return rows, {'increments': count,
            'max_force_residual_n': max(abs(a['force_residual_n']) for a in audits),
            'max_moment_residual_nm': max(abs(a['moment_residual_nm']) for a in audits),
            'minimum_tangent_eigenvalue_n_m': min(a['minimum_tangent_eigenvalue_n_m'] for a in audits),
            'max_accounted_change_j': max(abs(r['accounted_j']-initial) for r in rows)}

    def cycle(self, state, peak, maximum_drive_increment_m=4e-6):
        rows = []; audits = []
        for name, target in (('up', peak), ('down', np.zeros(len(self.springs)))):
            trace, audit = self.segment(state, target, maximum_drive_increment_m)
            rows.extend([dict(row, segment=name) for row in trace]); audits.append(audit)
        return rows, {'max_force_residual_n': max(a['max_force_residual_n'] for a in audits),
            'max_moment_residual_nm': max(a['max_moment_residual_nm'] for a in audits),
            'minimum_tangent_eigenvalue_n_m': min(a['minimum_tangent_eigenvalue_n_m'] for a in audits),
            'max_segment_accounted_change_j': max(a['max_accounted_change_j'] for a in audits)}


def three_zone_fixture(energy=2., spring=5000.):
    laws = [InterfaceLaw(area=.01/3, fracture_energy=energy) for _ in range(3)]
    return SpatialFixture(laws, [-.04, 0., .04], [-.04, .04], [spring, spring])
