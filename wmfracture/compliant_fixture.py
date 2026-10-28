"""Single apparatus: prescribed remote drive -> spring -> cohesive joint.

Small scalar feasibility model, not a general fracture simulator. All damage
history must be created by the same drive protocol; no prescribed joint gap.
"""
from dataclasses import asdict, dataclass, replace
import math

from wmfracture.cohesive import InterfaceLaw


@dataclass(frozen=True)
class Fixture:
    spring: float = 10000.  # N/m
    mass: float = .02
    damping: float = 18.   # N s/m

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()) or self.spring <= 0 or self.mass <= 0 or self.damping < 0:
            raise ValueError("Invalid compliant fixture")


@dataclass
class DriveState:
    gap: float = 0.
    velocity: float = 0.
    maximum_opening: float = 0.
    drive: float = 0.
    time: float = 0.
    step: int = 0
    actuator_work: float = 0.
    damping_heat: float = 0.


def validate_state(state):
    if not all(math.isfinite(v) for v in asdict(state).values()) or state.maximum_opening < max(state.gap, 0.)-1e-12:
        raise ValueError("Invalid physical fixture state")
    if state.time < 0 or type(state.step) is not int or state.step < 0 or state.damping_heat < -1e-14:
        raise ValueError("Invalid fixture clock/work state")


def energy(state, law, fixture):
    return (.5*fixture.mass*state.velocity**2+.5*fixture.spring*(state.drive-state.gap)**2
            +law.potential(state.gap, state.maximum_opening)+law.dissipated(state.maximum_opening))


def advance(state, law, fixture, next_drive, dt):
    """Damping splitting + velocity Verlet; exact work recorded for the drive."""
    omega = math.sqrt((fixture.spring+law.area*max(law.tensile_stiffness, law.compressive_stiffness))/fixture.mass)
    if not math.isfinite(next_drive) or not math.isfinite(dt) or not 0 < dt*omega < 1:
        raise ValueError("Drive and time step must be finite and resolved")
    decay = math.exp(-fixture.damping*dt/(2*fixture.mass))
    f0 = fixture.spring*(state.drive-state.gap)
    state.damping_heat += .5*fixture.mass*state.velocity**2*(1-decay**2)
    state.velocity *= decay
    state.velocity += .5*dt*(f0-law.area*law.traction(state.gap, state.maximum_opening))/fixture.mass
    state.gap += dt*state.velocity
    state.maximum_opening = max(state.maximum_opening,state.gap,0.)
    f1 = fixture.spring*(next_drive-state.gap)
    state.velocity += .5*dt*(f1-law.area*law.traction(state.gap, state.maximum_opening))/fixture.mass
    state.damping_heat += .5*fixture.mass*state.velocity**2*(1-decay**2)
    state.velocity *= decay
    state.actuator_work += .5*(f0+f1)*(next_drive-state.drive)
    state.drive = next_drive
    state.time += dt
    state.step += 1


def segment(state, law, fixture, target, duration, dt, sample_interval=.005):
    count, stride = round(duration/dt), round(sample_interval/dt)
    if count < 1 or stride < 1 or abs(count*dt-duration) > 1e-10 or abs(stride*dt-sample_interval) > 1e-10:
        raise ValueError("Segment clocks must align with dt")
    validate_state(state)
    start = state.drive
    initial = energy(state,law,fixture)+state.damping_heat-state.actuator_work
    maximum_residual = 0.
    first_failure = None
    rows = []
    for i in range(count+1):
        if i % stride == 0 or i == count:
            rows.append({**asdict(state), 'damage':law.damage(state.maximum_opening)})
            maximum_residual = max(maximum_residual,abs(energy(state,law,fixture)+state.damping_heat-state.actuator_work-initial))
        if first_failure is None and law.damage(state.maximum_opening) == 1.:
            first_failure = {'time':state.time,'drive':state.drive}
        if i < count:
            fraction = .5-.5*math.cos(math.pi*(i+1)/count)
            advance(state,law,fixture,start+(target-start)*fraction,dt)
    validate_state(state)
    return rows, {'maximum_energy_residual_j':maximum_residual,'first_failure':first_failure}


def cycle(state, law, fixture, peak, dt, ramp=1., hold=.25, rest=.5):
    """Remote-only cycle, including ramp down and rest; preserve the full state."""
    records, audits = [], []
    for name,target,duration in [('up',peak,ramp),('hold',peak,hold),('down',0.,ramp),('rest',0.,rest)]:
        rows,audit = segment(state,law,fixture,target,duration,dt)
        records.extend([{**row,'segment':name} for row in rows]); audits.append(audit)
    return records, {'maximum_segment_energy_residual_j':max(a['maximum_energy_residual_j'] for a in audits),
                     'first_failure':next((a['first_failure'] for a in audits if a['first_failure'] is not None),None)}


def nominal_remote_peak(law, fixture, joint_peak):
    """Quasistatic command only, NOT assignment of a hidden state or its proof."""
    if not math.isfinite(joint_peak) or not 0 <= joint_peak < law.failure_opening:
        raise ValueError("Preparation command must be below complete separation")
    return joint_peak+law.area*law.traction(joint_peak,joint_peak)/fixture.spring


def clone_state(state):
    validate_state(state)
    return replace(state)
