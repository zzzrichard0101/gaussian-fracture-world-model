from wmfracture.diagnostics import mechanical_checks


def test_freefall_check_excludes_contact_and_later_rebound():
    # A faster impact reaches the floor before the old fixed 0.05s cutoff.
    rows = [
        {"time": 0, "min_z": .626, "mean_vz": -10.0},
        {"time": .03, "min_z": .32, "mean_vz": -10.45},
        {"time": .04, "min_z": .216, "mean_vz": -10.6},
        {"time": .05, "min_z": .20, "mean_vz": -6.2},
        {"time": .20, "min_z": .7, "mean_vz": 4.0},
    ]
    for row in rows:
        row.update(Jp_changed_fraction=0.0, neighbor_stretch_gt2_fraction=0.0)
    checks = mechanical_checks(rows, -10.0, -15.0, .2, .03125)
    assert checks["precontact_frames_checked"] == 2
    assert checks["precontact_gravity_passed"]
    assert checks["floor_contact_observed"]
    assert not checks["fracture_accuracy_validated"]
