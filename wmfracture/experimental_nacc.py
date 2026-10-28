"""Probe-only correction of upstream NACC shear-branch hardening projection.

Upstream normalizes direction_p but not direction_q in a pressure/shear ray.
For beta=3e9 a captured finite trial state overflows its float32 discriminant.
Here the intended center-to-trial ray is solved in scaled coordinates, without
the discriminant or subtracting a huge center from a huge root. This changes
the upstream hardening update; it is not exact-source reproduction or a new
fracture law. All other return/stress calculations are retained upstream.

Installation replaces one module global in an isolated probe child process.
Checkpoints deliberately reject this experimental constitutive override.
"""
import importlib

import warp as wp

from mpm_solver_warp.mpm_solver_warp import MPMStateStruct, MPMModelStruct
from mpm_utils import kirchoff_stress_neoHookeanBoarden


@wp.func
def ray_pressure(p_trial: float, q_trial: float, p0: float, beta: float, M: float):
    center = (1.0 - beta) * p0 * 0.5
    half_width = (1.0 + beta) * p0 * 0.5
    direction_p = p_trial - center
    scale = wp.max(half_width, wp.max(wp.abs(direction_p), q_trial))
    dp = direction_p / scale
    dq = q_trial / scale
    a = M * M * dp * dp + (1.0 + 2.0 * beta) * dq * dq
    c = -M * M * (half_width / scale) * (half_width / scale)
    radius = wp.sqrt(-c / a)
    # f(trial)/scale^2 is evaluated directly, rather than subtracting a+c.
    y = (M * M * ((p_trial + beta * p0) / scale) * ((p_trial - p0) / scale)
         + (1.0 + 2.0 * beta) * dq * dq)
    # radius-1 = -f(trial)/(a*(radius+1)); avoid loss of precision near 1.
    return p_trial - ((y / a) / (radius + 1.0)) * direction_p


@wp.func
def corrected_return(F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int):
    # Adapted from the pinned upstream NonAssociativeCamClay_return_mapping.
    # Keep its arithmetic/branch classification and elastic projection together;
    # change only the shear hardening ray solution. Recomputing branch predicates
    # after calling upstream can disagree near yield at extreme beta.
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sigma = wp.vec3(0.0)
    wp.svd3(F_trial, U, sigma, V)
    logJp = state.particle_Jp[p]
    mu = model.mu[p]
    kappa = model.kappa[p]
    M = model.M
    beta = model.beta[p]
    sigma[0] = wp.max(sigma[0], 0.0)
    sigma[1] = wp.max(sigma[1], 0.0)
    sigma[2] = wp.max(sigma[2], 0.0)
    p0 = kappa * (0.00001 + wp.sinh(model.xi * wp.max(-logJp, 0.0)))
    J = sigma[0] * sigma[1] * sigma[2]
    b0 = sigma[0] * sigma[0]
    b1 = sigma[1] * sigma[1]
    b2 = sigma[2] * sigma[2]
    mean = (b0 + b1 + b2) / 3.0
    J_pow = wp.pow(J, -2.0 / 3.0)
    s0 = mu * J_pow * (b0 - mean)
    s1 = mu * J_pow * (b1 - mean)
    s2 = mu * J_pow * (b2 - mean)
    prime = kappa / 2.0 * (J - 1.0 / J)
    pressure = -prime * J
    coefficient = 1.5 * (1.0 + 2.0 * beta)
    y_pressure = M * M * (pressure + beta * p0) * (pressure - p0)
    s_norm_sq = s0 * s0 + s1 * s1 + s2 * s2
    y = coefficient * s_norm_sq + y_pressure
    f0 = sigma[0]
    f1 = sigma[1]
    f2 = sigma[2]
    next_logJp = logJp
    p_min = beta * p0
    if pressure > p0:
        Je_new = wp.sqrt(-2.0 * p0 / kappa + 1.0)
        f0 = wp.pow(Je_new, 1.0 / 3.0)
        f1 = f0
        f2 = f0
        if model.hardening > 0.5:
            next_logJp = logJp + wp.log(J / Je_new)
    elif pressure < -p_min:
        Je_new = wp.sqrt(2.0 * p_min / kappa + 1.0)
        f0 = wp.pow(Je_new, 1.0 / 3.0)
        f1 = f0
        f2 = f0
        if model.hardening > 0.5:
            next_logJp = logJp + wp.log(J / Je_new)
    elif y >= 1e-4:
        s_norm = wp.max(wp.sqrt(s_norm_sq), 1e-10)
        sqrt_factor = wp.sqrt(-y_pressure / coefficient)
        factor = wp.pow(J, 2.0 / 3.0) / mu * sqrt_factor / s_norm
        f0 = wp.sqrt(factor * s0 + mean)
        f1 = wp.sqrt(factor * s1 + mean)
        f2 = wp.sqrt(factor * s2 + mean)
        if model.hardening > 0.5 and p0 > 1e-4 and pressure < p0 - 1e-4 and pressure > 1e-4 - p_min:
            q = wp.sqrt(1.5) * s_norm
            p_fake = ray_pressure(pressure, q, p0, beta, M)
            Je_fake = wp.sqrt(wp.abs(1.0 - 2.0 * p_fake / kappa))
            if Je_fake > 1e-4:
                next_logJp = logJp + wp.log(J / Je_fake)
    state.particle_Jp[p] = next_logJp
    return U * wp.mat33(f0, 0.0, 0.0, 0.0, f1, 0.0, 0.0, 0.0, f2) * wp.transpose(V)


@wp.kernel
def compute_stress_corrected(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        elastic = corrected_return(state.particle_F_trial[p], state, model, p)
        state.particle_F[p] = elastic
        U = wp.mat33(0.0)
        V = wp.mat33(0.0)
        sigma = wp.vec3(0.0)
        wp.svd3(elastic, U, sigma, V)
        J = wp.determinant(elastic)
        stress = kirchoff_stress_neoHookeanBoarden(elastic, U, V, J, sigma,
            model.mu[p], model.lam[p], model.kappa[p], state)
        state.particle_stress[p] = (stress + wp.transpose(stress)) / 2.0


def install_for_probe(sim):
    if (sim.mpm_model.material.numpy() != 7).any():
        raise ValueError("Experimental NACC correction supports watermelon material only")
    module = importlib.import_module("mpm_solver_warp.mpm_solver_warp")
    original = module.compute_stress_from_F_trial
    module.compute_stress_from_F_trial = compute_stress_corrected
    sim._experimental_constitutive_override = "scaled-nacc-ray-hardening-v2"
    return original
