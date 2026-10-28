"""Conservative particle-space separation diagnostics, independent of appearance.

Connectivity measures spatially separated particle groups. It cannot infer
fracture surfaces in contact, material bonds, or physically correct cracks.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def persistent_motion_candidate(groups, relative_speed, min_duration=.04, min_speed=.1):
    top = groups[:2]
    return (len(top) == 2 and all(group["consecutive_duration"] >= min_duration - 1e-8 for group in top)
            and relative_speed is not None and relative_speed > min_speed)


def simultaneous_consensus(rows, initially_connected, minimum_votes=2):
    votes = {}
    for radius, frames in rows.items():
        if not initially_connected[radius]:
            continue
        for row in frames:
            if row["separation_candidate"]:
                votes[row["frame"]] = votes.get(row["frame"], 0) + 1
    return sorted(frame for frame, count in votes.items() if count >= minimum_votes)


def spatial_components(positions, volumes, velocities, radius, min_mass_fraction=.01):
    if radius <= 0 or min_mass_fraction <= 0:
        raise ValueError("Positive radius and mass threshold are required")
    positions, volumes, velocities = map(np.asarray, (positions, volumes, velocities))
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all() or (volumes <= 0).any():
        raise ValueError("Finite particles and positive volumes are required")
    pairs = cKDTree(positions).query_pairs(radius, output_type="ndarray").astype(np.int32)
    graph = csr_matrix((np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])), shape=(len(positions), len(positions)))
    count, labels = connected_components(graph, directed=False)
    mass = np.bincount(labels, weights=volumes, minlength=count)
    fractions = mass / mass.sum()
    order = np.argsort(-mass)
    groups = []
    # Constant density in current experiments: volume fractions equal mass fractions.
    for component in order[fractions[order] >= min_mass_fraction]:
        indices = np.flatnonzero(labels == component)
        weights = volumes[indices] / mass[component]
        groups.append({"component": int(component), "particles": int(len(indices)), "mass_fraction": float(fractions[component]),
                       "center": (positions[indices] * weights[:, None]).sum(axis=0).tolist(),
                       "velocity": (velocities[indices] * weights[:, None]).sum(axis=0).tolist()})
    gap, relative_speed = None, None
    if len(groups) >= 2:
        first = np.flatnonzero(labels == groups[0]["component"])
        second = np.flatnonzero(labels == groups[1]["component"])
        gap = float(cKDTree(positions[first]).query(positions[second])[0].min())
        relative_speed = float(np.linalg.norm(np.array(groups[0]["velocity"]) - groups[1]["velocity"]))
    return labels, {"radius": float(radius), "edge_count": len(pairs), "components_including_dust": int(count),
                    "major_components": groups, "largest_mass_fraction": float(fractions.max()),
                    "dust_mass_fraction": float(fractions[fractions < min_mass_fraction].sum()),
                    "two_largest_gap": gap, "two_largest_relative_speed": relative_speed}


class ComponentTracker:
    """Associate major groups by particle-ID mass IoU, not component numbering."""
    def __init__(self, volumes, min_iou=.5):
        self.volumes = np.asarray(volumes)
        self.previous = np.zeros(len(volumes), dtype=np.int32)
        self.tracks = {}
        self.next_id = 1
        self.min_iou = min_iou

    def update(self, labels, groups, timestamp):
        current = np.zeros(len(labels), dtype=np.int32)
        updated = {}
        used = set()
        for group in groups:
            mask = labels == group["component"]
            current_mass = self.volumes[mask].sum()
            best_id, best_iou = 0, 0.
            for identifier, previous_track in self.tracks.items():
                if identifier in used:
                    continue
                previous_mask = self.previous == identifier
                intersection = self.volumes[mask & previous_mask].sum()
                union = current_mass + self.volumes[previous_mask].sum() - intersection
                iou = intersection / union
                if iou > best_iou:
                    best_id, best_iou = identifier, iou
            if best_iou > self.min_iou:
                first_time = self.tracks[best_id]["first_time"]
            else:
                best_id = self.next_id
                self.next_id += 1
                first_time = timestamp
            used.add(best_id)
            current[mask] = best_id
            group.update(track_id=best_id, consecutive_duration=float(timestamp - first_time), mass_iou=float(best_iou))
            updated[best_id] = {"first_time": first_time}
        self.previous, self.tracks = current, updated
        return current
