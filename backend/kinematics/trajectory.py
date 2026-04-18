"""
Trajectory Tracking for 3-DOF Robotic Manipulators

Samples a user-defined path at discrete points, runs IK at each point,
checks full workspace feasibility, and returns joint variable sequences.
"""

import numpy as np
from kinematics.inverse import (
    inverse_kinematics_RRR,
    inverse_kinematics_RRR_numerical,
    inverse_kinematics_PPP,
)


def generate_line_trajectory(
    start: list, end: list, num_points: int = 50
) -> list:
    """Generate evenly-spaced points along a straight line."""
    return [
        [
            start[0] + t * (end[0] - start[0]),
            start[1] + t * (end[1] - start[1]),
            start[2] + t * (end[2] - start[2]),
        ]
        for t in np.linspace(0, 1, num_points)
    ]


def generate_circle_trajectory(
    center: list, radius: float, num_points: int = 72, plane: str = "xy"
) -> list:
    """Generate points along a circular path in the XY, XZ, or YZ plane."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = []
    cx, cy, cz = center
    for a in angles:
        if plane == "xy":
            points.append([cx + radius * np.cos(a), cy + radius * np.sin(a), cz])
        elif plane == "xz":
            points.append([cx + radius * np.cos(a), cy, cz + radius * np.sin(a)])
        elif plane == "yz":
            points.append([cx, cy + radius * np.cos(a), cz + radius * np.sin(a)])
    return points


def track_trajectory_RRR(
    path_points: list,
    link_lengths: list,
    joint_limits: list = None,
    elbow_up: bool = True,
    numerical_fallback: bool = True,
) -> dict:
    """
    Run IK at every point along a trajectory for an RRR manipulator.

    Args:
        path_points:    list of [x, y, z] target positions
        link_lengths:   [a1, a2, a3]
        joint_limits:   [(min, max)] * 3 in radians, default ±π
        elbow_up:       preferred IK configuration
        numerical_fallback: use SciPy IK if analytical fails

    Returns:
        dict with per-point results, feasibility summary, and joint sequences
    """
    results = []
    failed_indices = []
    joint_seq = [[], [], []]   # theta1, theta2, theta3 over time
    prev_angles = None

    for i, point in enumerate(path_points):
        x, y, z = point

        # Try analytical first
        ik = inverse_kinematics_RRR(x, y, z, link_lengths, elbow_up, joint_limits)

        # Fallback to numerical if analytical fails
        if not ik.success and numerical_fallback:
            guess = prev_angles if prev_angles else None
            ik = inverse_kinematics_RRR_numerical(x, y, z, link_lengths, guess, joint_limits)

        if ik.success:
            prev_angles = ik.joint_values
            for j in range(3):
                joint_seq[j].append(float(ik.joint_values[j]))
        else:
            failed_indices.append(i)
            for j in range(3):
                joint_seq[j].append(None)

        results.append({
            "point_index": i,
            "target": {"x": float(x), "y": float(y), "z": float(z)},
            "success": ik.success,
            "joint_values": ik.joint_values if ik.success else [],
            "joint_values_deg": ik.joint_values_deg if ik.success else [],
            "error": float(ik.error) if ik.success else None,
            "configuration": ik.configuration,
            "message": ik.message,
        })

    total = len(path_points)
    feasible = total - len(failed_indices)
    is_feasible = len(failed_indices) == 0

    return {
        "type": "RRR",
        "is_feasible": is_feasible,
        "total_points": total,
        "feasible_points": feasible,
        "failed_points": len(failed_indices),
        "failed_indices": failed_indices,
        "feasibility_ratio": round(feasible / total, 4),
        "joint_sequences": {
            "theta1": joint_seq[0],
            "theta2": joint_seq[1],
            "theta3": joint_seq[2],
        },
        "path_points": [[float(p[0]), float(p[1]), float(p[2])] for p in path_points],
        "per_point_results": results,
        "link_lengths": link_lengths,
    }


def track_trajectory_PPP(
    path_points: list,
    joint_limits: list = None,
) -> dict:
    """
    Run IK at every point along a trajectory for a PPP manipulator.

    For Cartesian robots, displacements directly equal the target coordinates.
    """
    results = []
    failed_indices = []
    joint_seq = [[], [], []]

    for i, point in enumerate(path_points):
        x, y, z = point
        ik = inverse_kinematics_PPP(x, y, z, joint_limits)

        if ik.success:
            for j in range(3):
                joint_seq[j].append(float(ik.joint_values[j]))
        else:
            failed_indices.append(i)
            for j in range(3):
                joint_seq[j].append(None)

        results.append({
            "point_index": i,
            "target": {"x": float(x), "y": float(y), "z": float(z)},
            "success": ik.success,
            "joint_values": ik.joint_values if ik.success else [],
            "error": 0.0,
            "message": ik.message,
        })

    total = len(path_points)
    feasible = total - len(failed_indices)

    return {
        "type": "PPP",
        "is_feasible": len(failed_indices) == 0,
        "total_points": total,
        "feasible_points": feasible,
        "failed_points": len(failed_indices),
        "failed_indices": failed_indices,
        "feasibility_ratio": round(feasible / total, 4),
        "joint_sequences": {
            "d1": joint_seq[0],
            "d2": joint_seq[1],
            "d3": joint_seq[2],
        },
        "path_points": [[float(p[0]), float(p[1]), float(p[2])] for p in path_points],
        "per_point_results": results,
    }


def compute_workspace_boundary(
    link_lengths: list,
    num_samples: int = 500,
) -> dict:
    """
    Sample the reachable workspace of an RRR manipulator.
    Returns a point cloud of reachable positions.
    """
    from kinematics.forward import forward_kinematics_RRR

    points = []
    angles = np.linspace(-np.pi, np.pi, num_samples // 10)

    for t1 in angles:
        for t2 in angles:
            for t3 in np.linspace(-np.pi, np.pi, 5):
                fk = forward_kinematics_RRR([t1, t2, t3], link_lengths)
                ee = fk["end_effector"]
                points.append([round(ee["x"], 4), round(ee["y"], 4)])

    max_reach = sum(link_lengths)
    min_reach = abs(link_lengths[0] - link_lengths[1]) - link_lengths[2]
    min_reach = max(0, min_reach)

    return {
        "max_reach": max_reach,
        "min_reach": min_reach,
        "workspace_points": points[:2000],  # cap for API response size
        "link_lengths": link_lengths,
    }


if __name__ == "__main__":
    print("=== Line trajectory (RRR) ===")
    path = generate_line_trajectory([0.5, 0.0, 0.0], [1.2, 0.8, 0.0], num_points=10)
    result = track_trajectory_RRR(path, link_lengths=[1.0, 0.8, 0.5])
    print(f"Feasible: {result['is_feasible']}")
    print(f"Points: {result['feasible_points']}/{result['total_points']} reachable")

    print("\n=== Circle trajectory (RRR) ===")
    circle = generate_circle_trajectory([0.0, 0.0, 0.0], radius=1.5, num_points=20)
    result2 = track_trajectory_RRR(circle, link_lengths=[1.0, 0.8, 0.5])
    print(f"Feasible: {result2['is_feasible']}")
    print(f"Points: {result2['feasible_points']}/{result2['total_points']} reachable")
    print(f"Failed indices: {result2['failed_indices']}")

    print("\n=== Out-of-reach trajectory (RRR) ===")
    far_path = generate_line_trajectory([2.5, 0.0, 0.0], [3.0, 0.5, 0.0], num_points=5)
    result3 = track_trajectory_RRR(far_path, link_lengths=[1.0, 0.8, 0.5])
    print(f"Feasible: {result3['is_feasible']}")
    print(f"Points: {result3['feasible_points']}/{result3['total_points']} reachable")

    print("\n=== PPP trajectory ===")
    ppp_path = generate_line_trajectory([0.1, 0.1, 0.1], [1.0, 1.0, 1.0], num_points=5)
    result4 = track_trajectory_PPP(ppp_path, joint_limits=[(0, 2.0)] * 3)
    print(f"Feasible: {result4['is_feasible']}")
    print(f"d1 sequence: {result4['joint_sequences']['d1']}")
