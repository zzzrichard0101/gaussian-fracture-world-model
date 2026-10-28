"""Reduced mode-I cohesive/contact fixture for two rigid blocks, in SI units.

Standard bilinear traction-separation with irreversible maximum opening and
an independent compressive penalty. This is a prescribed interface and one
translational DOF per block, not a 3D bulk-fracture solver or Abaqus/MOOSE port.
Analytical energy and collision solutions make implementation checks possible.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class InterfaceLaw:
    area: float = .001
    tensile_stiffness: float = 1e7  # Pa/m
    compressive_stiffness: float = 1e8  # Pa/m, finite penalty
    peak_traction: float = 1000.  # Pa
    fracture_energy: float = 2.  # J/m^2

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Cohesive parameters must be finite and positive")
        if self.failure_opening <= self.initiation_opening:
            raise ValueError("Failure opening must exceed damage initiation opening")

    @property
    def initiation_opening(self):
        return self.peak_traction / self.tensile_stiffness

    @property
    def failure_opening(self):
        return 2 * self.fracture_energy / self.peak_traction

    def damage(self, maximum_opening):
        onset, final = self.initiation_opening, self.failure_opening
        if maximum_opening <= onset:
            return 0.
        if maximum_opening >= final:
            return 1.
        return final * (maximum_opening - onset) / (maximum_opening * (final - onset))

    def traction(self, opening, maximum_opening):
        if opening < 0:
            return self.compressive_stiffness * opening
        return (1 - self.damage(maximum_opening)) * self.tensile_stiffness * opening

    def potential(self, opening, maximum_opening):
        stiffness = (self.compressive_stiffness if opening < 0 else
                     (1 - self.damage(maximum_opening)) * self.tensile_stiffness)
        return .5 * self.area * stiffness * opening ** 2

    def dissipated(self, maximum_opening):
        fraction = (maximum_opening - self.initiation_opening) / (self.failure_opening - self.initiation_opening)
        return self.area * self.fracture_energy * min(1., max(0., fraction))


@dataclass
class InterfaceState:
    x_left: float
    x_right: float
    v_left: float = 0.
    v_right: float = 0.
    maximum_opening: float = 0.
    time: float = 0.
    step_index: int = 0
    external_work: float = 0.


@dataclass
class TwoBlockSystem:
    law: InterfaceLaw
    state: InterfaceState
    mass_left: float = .1
    mass_right: float = .1
    width: float = .1

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in (self.mass_left, self.mass_right, self.width)):
            raise ValueError("Masses and block width must be positive")
        self.validate()

    @property
    def opening(self):
        return self.state.x_right - self.state.x_left - self.width

    @property
    def reduced_mass(self):
        return self.mass_left * self.mass_right / (self.mass_left + self.mass_right)

    def validate(self):
        state = self.state
        if (any(not math.isfinite(value) for value in asdict(state).values())
            or state.maximum_opening < 0 or state.time < 0
            or type(state.step_index) is not int or state.step_index < 0):
            raise ValueError("Invalid cohesive state")
        if self.opening > state.maximum_opening + 1e-12:
            raise ValueError("Maximum opening must include current positive opening")

    def energies(self):
        state = self.state
        kinetic = .5 * (self.mass_left * state.v_left ** 2 + self.mass_right * state.v_right ** 2)
        potential = self.law.potential(self.opening, state.maximum_opening)
        dissipated = self.law.dissipated(state.maximum_opening)
        return {"kinetic": kinetic, "potential": potential, "dissipated": dissipated,
                "accounted_total": kinetic + potential + dissipated}

    def momentum(self):
        return self.mass_left * self.state.v_left + self.mass_right * self.state.v_right

    def impulse(self, left=0., right=0.):
        """External velocity increments; exact work is recorded, no state reset."""
        if not all(math.isfinite(x) for x in (left, right)):
            raise ValueError("Nonfinite impulse")
        before = self.energies()["kinetic"]
        self.state.v_left += left
        self.state.v_right += right
        work = self.energies()["kinetic"] - before
        self.state.external_work += work
        return {"time": self.state.time, "step": self.state.step_index, "kinetic_work": work,
                "momentum_increment": self.mass_left * left + self.mass_right * right}

    def advance(self, dt):
        """Velocity Verlet; dt is bounded by the two-mass linear stability limit."""
        omega = math.sqrt(self.law.area * max(self.law.tensile_stiffness, self.law.compressive_stiffness) / self.reduced_mass)
        if not math.isfinite(dt) or not 0 < dt * omega < 1.:
            raise ValueError("dt must resolve the highest two-block natural frequency")
        state = self.state
        force = self.law.area * self.law.traction(self.opening, state.maximum_opening)
        state.v_left += .5 * dt * force / self.mass_left
        state.v_right -= .5 * dt * force / self.mass_right
        state.x_left += dt * state.v_left
        state.x_right += dt * state.v_right
        state.maximum_opening = max(state.maximum_opening, self.opening, 0.)
        force = self.law.area * self.law.traction(self.opening, state.maximum_opening)
        state.v_left += .5 * dt * force / self.mass_left
        state.v_right -= .5 * dt * force / self.mass_right
        state.time += dt
        state.step_index += 1
        self.validate()

    def prescribe_opening(self, opening, relative_velocity=0.):
        """Boundary-controlled loading; preserve mass COM, track actuator work.

        Work is integrated trapezoidally along the supplied displacement path.
        Used only while actuators prescribe the two rigid translations, then
        released with the velocities of that path. No latent state is erased.
        """
        if not all(math.isfinite(value) for value in (opening, relative_velocity)):
            raise ValueError("Nonfinite prescribed motion")
        state = self.state
        old_gap = self.opening
        old_force = self.law.area * self.law.traction(old_gap, state.maximum_opening)
        mass = self.mass_left + self.mass_right
        center = (self.mass_left * state.x_left + self.mass_right * state.x_right) / mass
        center_velocity = self.momentum() / mass
        kinetic_before = self.energies()["kinetic"]
        state.x_left = center - self.mass_right / mass * (self.width + opening)
        state.x_right = center + self.mass_left / mass * (self.width + opening)
        state.v_left = center_velocity - self.mass_right / mass * relative_velocity
        state.v_right = center_velocity + self.mass_left / mass * relative_velocity
        state.maximum_opening = max(state.maximum_opening, opening, 0.)
        new_force = self.law.area * self.law.traction(opening, state.maximum_opening)
        state.external_work += (.5 * (old_force + new_force) * (opening - old_gap)
                                + self.energies()["kinetic"] - kinetic_before)
        self.validate()


def make_system(law=None):
    return TwoBlockSystem(law or InterfaceLaw(), InterfaceState(-.05, .05))


def prescribed_cycle(system, peak, samples=2000, duration=1.):
    """Smooth displacement-controlled load/unload cycle, released at rest."""
    if not math.isfinite(peak) or peak < 0 or samples < 2 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid loading cycle")
    records = []
    start_time, start_step = system.state.time, system.state.step_index
    for index in range(samples + 1):
        # sin^2 ensures zero opening and zero velocity at both endpoints.
        phase = index / samples
        gap = peak * math.sin(math.pi * phase) ** 2
        velocity = peak * math.pi * math.sin(2 * math.pi * phase) / duration
        if index in (0, samples):
            gap, velocity = 0., 0.
        system.prescribe_opening(gap, velocity)
        system.state.time = start_time + duration * phase
        system.state.step_index = start_step + index
        if index % max(1, samples // 100) == 0:
            records.append({"phase": phase, "time": system.state.time, "opening": system.opening,
                "maximum_opening": system.state.maximum_opening,
                "damage": system.law.damage(system.state.maximum_opening),
                "traction": system.law.traction(system.opening, system.state.maximum_opening)})
    return records


def save_state(path, system):
    """Dedicated reduced-fixture JSON state, independent of MPM checkpoint schema."""
    path = Path(path)
    system.validate()
    payload = {"schema": "cohesive-two-block-1", "law": asdict(system.law), "state": asdict(system.state),
               "mass_left": system.mass_left, "mass_right": system.mass_right, "width": system.width}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    envelope = {"payload": payload, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(envelope, stream, indent=2, allow_nan=False)


def load_state(path):
    envelope = json.loads(Path(path).read_text())
    payload = envelope["payload"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if (payload["schema"] != "cohesive-two-block-1"
        or hashlib.sha256(canonical.encode()).hexdigest() != envelope["sha256"]):
        raise ValueError("Unsupported/corrupt reduced cohesive checkpoint")
    return TwoBlockSystem(InterfaceLaw(**payload["law"]), InterfaceState(**payload["state"]),
                          payload["mass_left"], payload["mass_right"], payload["width"])
