"""Controlled beam and Gaussian-to-control encoders; no learned renderer or inverse physics."""
import numpy as np
from scipy.linalg import expm


class Beam:
    def __init__(self, nodes=33, bending=.002, damping=1.5, dt=.025):
        if nodes < 7 or min(bending, dt) <= 0 or damping < 0:
            raise ValueError('Positive stable beam settings required')
        self.x = np.linspace(0., 1., nodes); self.free = np.arange(2, nodes-2)
        h = self.x[1]-self.x[0]; n = len(self.free)
        second = np.diff(np.eye(nodes), n=2, axis=0)/h**2
        self.stiffness = bending*second[:, self.free].T@second[:, self.free]
        system = np.block([[np.zeros((n,n)), np.eye(n)], [-self.stiffness, -damping*np.eye(n)]])
        augmented = np.zeros((3*n,3*n)); augmented[:2*n,:2*n] = system
        augmented[n:2*n,2*n:] = np.eye(n)
        evolution = expm(augmented*dt)
        self.transition = evolution[:2*n,:2*n]; self.forcing = evolution[:2*n,2*n:]
        self.h, self.dt = h, dt

    def force(self, center, total, width):
        profile = np.exp(-.5*((self.x[self.free]-center)/width)**2)
        return profile/profile.sum()*total/self.h

    def step(self, state, force):
        return np.asarray(state)@self.transition.T+np.asarray(force)@self.forcing.T

    def full(self, state):
        state = np.asarray(state); n = len(self.free)
        result = np.zeros((*state.shape[:-1],len(self.x),2))
        result[...,self.free,0] = state[...,:n]; result[...,self.free,1] = state[...,n:]
        return result


def interpolation(source, query):
    source, query = np.asarray(source), np.asarray(query)
    if len(source) < 2 or np.any(np.diff(source) <= 0):
        raise ValueError('Strictly increasing source coordinates required')
    out = np.zeros((len(query),len(source)))
    right = np.clip(np.searchsorted(source,query),1,len(source)-1); left = right-1
    weight = np.clip((query-source[left])/(source[right]-source[left]),0.,1.)
    out[np.arange(len(query)),left] = 1-weight; out[np.arange(len(query)),right] += weight
    return out


def gaussian_layout(spec):
    count, kind, support, seed = spec
    axis = (np.arange(count)+.5)/count
    if kind == 'left': axis = axis**2
    elif kind == 'right': axis = 1-(1-axis)**2
    elif kind == 'jitter': axis += np.random.default_rng(seed).uniform(-.3,.3,count)/count
    elif kind != 'uniform': raise ValueError('Unknown layout')
    edges = np.r_[0.,(axis[:-1]+axis[1:])/2,1.]
    area = np.diff(edges)*.03
    sigma_x, sigma_y = .018*support, .025
    # This is an appearance amplitude, never a physical mass or force.
    opacity = 1.2*area/(2*np.pi*sigma_x*sigma_y)
    return {'x':axis,'area_proxy':area,'sigma_x':sigma_x,'sigma_y':sigma_y,
        'opacity':np.clip(opacity,1e-5,.95),'opacity_scale':1.,'kind':kind,'seed':seed}


def encoder_matrix(layout, physical_x, anchors, method, width=.035, density_bandwidth=.035):
    x = layout['x']; sample = interpolation(physical_x,x)
    if method == 'grid_aug': return interpolation(x,anchors)@sample
    variance = width**2+(layout['sigma_x']**2 if method.startswith('kernel') else 0.)
    weights = np.exp(-.5*(anchors[:,None]-x[None])**2/variance)
    if method == 'density_aug':
        density = np.exp(-.5*((x[:,None]-x[None])/density_bandwidth)**2).sum(axis=1)
        weights /= density[None]
    elif method in ('weight_aug','kernel_aug','kernel_consistency'):
        # Observable appearance amplitudes and kernel support define a proxy
        # integration measure; its equality to material area is not assumed.
        measure = layout['opacity']*layout['sigma_x']*layout['sigma_y']
        weights *= measure[None]
    elif method not in ('point','point_aug'):
        raise ValueError('Unknown encoder')
    weights /= weights.sum(axis=1,keepdims=True)
    return weights@sample


def episodes(beam, count, seed, centers, config):
    rng = np.random.default_rng(seed); n = len(beam.free); frames = config['frames']
    _, modes = np.linalg.eigh(beam.stiffness)
    states, forces, commands = [], [], []
    for _ in range(count):
        state = np.r_[modes[:,:4]@rng.normal(0,.004,4), modes[:,:4]@rng.normal(0,.02,4)]
        trajectory = [state.copy()]; inputs = []; controls = []
        for step in range(frames):
            if step % config['action_hold_frames'] == 0:
                center = float(rng.choice(centers)); total = float(rng.uniform(-1,1)*config['maximum_total_force'])
            force = beam.force(center,total,config['force_width'])
            state = beam.step(state,force); trajectory.append(state.copy()); inputs.append(force); controls.append([center,total])
        states.append(trajectory); forces.append(inputs); commands.append(controls)
    return {'states':np.asarray(states),'forces':np.asarray(forces),'commands':np.asarray(commands)}


def normalized_features(beam, full_state, forces, matrix, config):
    current = np.einsum('ij,...jk->...ik',matrix,full_state)
    current = current[...,beam.free,:]/np.asarray(config['state_scales'])
    return np.concatenate((current.reshape(*current.shape[:-2],-1),forces/config['force_scale']),axis=-1)


def normalized_state(beam, full_state, config):
    return (full_state[...,beam.free,:]/np.asarray(config['state_scales'])).reshape(*full_state.shape[:-2],-1)


def decode_state(beam, normalized, config):
    value = np.asarray(normalized).reshape(*np.asarray(normalized).shape[:-1],len(beam.free),2)*config['state_scales']
    return np.concatenate((value[...,0],value[...,1]),axis=-1)
