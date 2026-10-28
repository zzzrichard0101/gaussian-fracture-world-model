"""Mechanical sanity checks; these do not measure fracture accuracy."""


def mechanical_checks(metrics, initial_vz, gravity_z, floor_height, dx):
    pre_contact = []
    for row in metrics:
        # Quadratic MPM interpolation touches the plane before particle centers
        # reach it. Stop at first possible contact, also excluding later rebound.
        if row["min_z"] <= floor_height + 1.5 * dx:
            break
        pre_contact.append(row)
    velocity_error = max((abs(row["mean_vz"] - initial_vz - gravity_z * row["time"])
                          for row in pre_contact), default=None)
    minimum_z = min(row["min_z"] for row in metrics)
    return {
        "precontact_frames_checked": len(pre_contact),
        "precontact_gravity_vz_max_error": velocity_error,
        "precontact_gravity_passed": velocity_error is not None and velocity_error < 0.005,
        "minimum_particle_z": minimum_z,
        "floor_penetration_less_than_one_cell": minimum_z >= floor_height - dx,
        "floor_contact_observed": minimum_z < floor_height + dx,
        "max_history_changed_fraction": max(row["Jp_changed_fraction"] for row in metrics),
        "max_neighbor_stretch_gt2_fraction": max(row["neighbor_stretch_gt2_fraction"] for row in metrics),
        "fracture_accuracy_validated": False,
    }
