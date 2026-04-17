"""
Trajectory tracking for RRR and PPP manipulators.

- Samples a user-defined path at discrete points
- Calls IK at each point to get joint variables
- Checks workspace feasibility for the full trajectory
- Returns joint angle sequences for animation
"""

import numpy as np
from .inverse import ik_RRR, ik_PPP
from .forward import fk_RRR, fk_PPP


# ---------------------------------------------------------------------------
# Path generators
# ---------------------------------------------------------------------------

def line_path(start, end, n_points=50):
    """Linearly interpolated path between two 2D/3D points."""
    start, end = np.array(start, dtype=float), np.array(end, dtype=float)
    return [start + t * (end - start) for t in np.linspace(0, 1, n_points)]


def circle_path(center, radius, z=0.0, n_points=72, angle_start=0, angle_end=2 * np.pi):
    """Circular arc in the XY plane."""
    cx, cy = center[:2]
    angles = np.linspace(angle_start, angle_end, n_points)
    return [[cx + radius * np.cos(a), cy + radius * np.sin(a), z] for a in angles]


def custom_path(points):
    """Use an explicit list of [x, y] or [x, y, phi] waypoints as-is."""
    return [np.array(p, dtype=float) for p in points]


# ---------------------------------------------------------------------------
# Workspace boundary (RRR)
# ---------------------------------------------------------------------------

def workspace_boundary_RRR(link_lengths, n_samples=360):
    """
    Compute the approximate workspace boundary for the first two links of a
    planar RRR manipulator (ignoring L3 orientation, just the reachable annulus).

    Returns a dict with outer and inner radius, and sample boundary points.
    """
    L1, L2, L3 = link_lengths
    r_max = L1 + L2   # wrist-centre reachable radius
    r_min = abs(L1 - L2)

    angles = np.linspace(0, 2 * np.pi, n_samples)
    outer = [[r_max * np.cos(a), r_max * np.sin(a)] for a in angles]
    inner = [[r_min * np.cos(a), r_min * np.sin(a)] for a in angles]

    return {
        "r_max": float(r_max),
        "r_min": float(r_min),
        "outer_boundary": outer,
        "inner_boundary": inner,
    }


# ---------------------------------------------------------------------------
# Trajectory tracker
# ---------------------------------------------------------------------------

def track_trajectory_RRR(path_points, link_lengths, elbow="up", phi=0.0):
    """
    Run IK along every point in path_points for a 3-DOF planar RRR arm.

    Args:
        path_points  : list of [x, y] or [x, y, phi] points
        link_lengths : [L1, L2, L3]
        elbow        : "up" or "down"
        phi          : default end-effector orientation (used if point has no phi)

    Returns:
        dict with:
            feasible        : bool  (True only if ALL points are reachable)
            reachable_mask  : list of bool per point
            joint_sequence  : list of [t1, t2, t3] per point (None if unreachable)
            positions       : list of [x, y] (FK-verified)
            infeasible_pts  : indices where IK failed
            summary         : human-readable string
    """
    joint_sequence  = []
    reachable_mask  = []
    positions       = []
    infeasible_pts  = []

    prev_angles = None

    for i, pt in enumerate(path_points):
        pt = np.array(pt, dtype=float)

        # Allow per-point orientation override
        if pt.shape[0] >= 3:
            target = pt.tolist()
        else:
            target = [pt[0], pt[1], phi]

        result = ik_RRR(target, link_lengths, elbow=elbow)

        if not result["reachable"] and prev_angles is not None:
            # Try numerical fallback with previous angles as initial guess
            from .inverse import ik_RRR_numerical
            result = ik_RRR_numerical(target, link_lengths,
                                      initial_guess=prev_angles)

        if result["reachable"]:
            angles = result["joint_angles"]
            fk = fk_RRR(angles, link_lengths)
            joint_sequence.append(angles)
            positions.append(fk["position"][:2])
            reachable_mask.append(True)
            prev_angles = angles
        else:
            joint_sequence.append(None)
            positions.append(None)
            reachable_mask.append(False)
            infeasible_pts.append(i)

    feasible = all(reachable_mask)
    pct = 100 * sum(reachable_mask) / len(reachable_mask)

    if feasible:
        summary = f"Trajectory fully feasible — all {len(path_points)} points reachable."
    else:
        summary = (
            f"Trajectory partially infeasible — "
            f"{len(infeasible_pts)}/{len(path_points)} points unreachable "
            f"({pct:.1f}% reachable). "
            f"Consider adjusting link lengths."
        )

    return {
        "feasible":       feasible,
        "reachable_mask": reachable_mask,
        "joint_sequence": joint_sequence,
        "positions":      positions,
        "infeasible_pts": infeasible_pts,
        "summary":        summary,
    }


def track_trajectory_PPP(path_points, axis_directions=None):
    """
    Run IK along every point for a 3-DOF PPP (Cartesian) manipulator.
    Always fully feasible as there are no workspace limits (unbounded).

    Returns equivalent structure to track_trajectory_RRR.
    """
    joint_sequence = []
    positions      = []

    for pt in path_points:
        result = ik_PPP(pt, axis_directions)
        fk     = fk_PPP(result["joint_displacements"], axis_directions)
        joint_sequence.append(result["joint_displacements"])
        positions.append(fk["position"])

    n = len(path_points)
    return {
        "feasible":       True,
        "reachable_mask": [True] * n,
        "joint_sequence": joint_sequence,
        "positions":      positions,
        "infeasible_pts": [],
        "summary":        f"PPP trajectory fully feasible — all {n} points reachable.",
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== RRR line trajectory test ===")
    L = [1.0, 1.0, 0.5]
    path = line_path([0.8, 0.2], [1.5, 0.5], n_points=20)
    res = track_trajectory_RRR(path, L)
    print(f"  {res['summary']}")
    if res["feasible"]:
        print(f"  First joint set : {[round(v, 4) for v in res['joint_sequence'][0]]}")
        print(f"  Last  joint set : {[round(v, 4) for v in res['joint_sequence'][-1]]}")

    print("\n=== RRR circle trajectory test ===")
    path2 = circle_path([0.8, 0.0], radius=0.5, n_points=36)
    res2 = track_trajectory_RRR(path2, L)
    print(f"  {res2['summary']}")

    print("\n=== RRR unreachable trajectory test ===")
    path3 = line_path([2.0, 0.0], [3.0, 0.0], n_points=10)
    res3 = track_trajectory_RRR(path3, L)
    print(f"  {res3['summary']}")

    print("\n=== PPP trajectory test ===")
    path4 = line_path([0, 0, 0], [1, 2, 3], n_points=10)
    res4 = track_trajectory_PPP(path4)
    print(f"  {res4['summary']}")
