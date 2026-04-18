"""
Trajectory Tracking for 3-DOF Robotic Manipulators
"""

import numpy as np
from kinematics.inverse import (
    inverse_kinematics_RRR,
    inverse_kinematics_RRR_numerical,
    inverse_kinematics_PPP,
)


def to_python(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_line_trajectory(start, end, num_points=50):
    return [
        [
            start[0] + t * (end[0] - start[0]),
            start[1] + t * (end[1] - start[1]),
            start[2] + t * (end[2] - start[2]),
        ]
        for t in np.linspace(0, 1, num_points)
    ]


def generate_circle_trajectory(center, radius, num_points=72, plane="xy"):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    cx, cy, cz = center
    points = []
    for a in angles:
        if plane == "xy":
            points.append([cx + radius * np.cos(a), cy + radius * np.sin(a), cz])
        elif plane == "xz":
            points.append([cx + radius * np.cos(a), cy, cz + radius * np.sin(a)])
        elif plane == "yz":
            points.append([cx, cy + radius * np.cos(a), cz + radius * np.sin(a)])
    return points


def track_trajectory_RRR(path_points, link_lengths, joint_limits=None, elbow_up=True, numerical_fallback=True):
    results = []
    failed_indices = []
    joint_seq = [[], [], []]
    prev_angles = None

    for i, point in enumerate(path_points):
        x, y, z = float(point[0]), float(point[1]), float(point[2])

        ik = inverse_kinematics_RRR(x, y, z, link_lengths, elbow_up, joint_limits)

        if not ik.success and numerical_fallback:
            ik = inverse_kinematics_RRR_numerical(x, y, z, link_lengths, prev_angles, joint_limits)

        if ik.success:
            prev_angles = ik.joint_values
            for j in range(3):
                joint_seq[j].append(float(ik.joint_values[j]))
        else:
            failed_indices.append(int(i))
            for j in range(3):
                joint_seq[j].append(None)

        results.append({
            "point_index": int(i),
            "target": {"x": float(x), "y": float(y), "z": float(z)},
            "success": bool(ik.success),
            "joint_values": [float(v) for v in ik.joint_values] if ik.success else [],
            "joint_values_deg": [float(v) for v in ik.joint_values_deg] if ik.success else [],
            "error": float(ik.error) if ik.success else None,
            "configuration": str(ik.configuration),
            "message": str(ik.message),
        })

    total = len(path_points)
    feasible = total - len(failed_indices)

    return to_python({
        "type": "RRR",
        "is_feasible": bool(len(failed_indices) == 0),
        "total_points": int(total),
        "feasible_points": int(feasible),
        "failed_points": int(len(failed_indices)),
        "failed_indices": [int(i) for i in failed_indices],
        "feasibility_ratio": float(round(feasible / total, 4)),
        "joint_sequences": {
            "theta1": joint_seq[0],
            "theta2": joint_seq[1],
            "theta3": joint_seq[2],
        },
        "path_points": [[float(p[0]), float(p[1]), float(p[2])] for p in path_points],
        "per_point_results": results,
        "link_lengths": link_lengths,
    })


def track_trajectory_PPP(path_points, joint_limits=None):
    results = []
    failed_indices = []
    joint_seq = [[], [], []]

    for i, point in enumerate(path_points):
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        ik = inverse_kinematics_PPP(x, y, z, joint_limits)

        if ik.success:
            for j in range(3):
                joint_seq[j].append(float(ik.joint_values[j]))
        else:
            failed_indices.append(int(i))
            for j in range(3):
                joint_seq[j].append(None)

        results.append({
            "point_index": int(i),
            "target": {"x": float(x), "y": float(y), "z": float(z)},
            "success": bool(ik.success),
            "joint_values": [float(v) for v in ik.joint_values] if ik.success else [],
            "error": 0.0,
            "message": str(ik.message),
        })

    total = len(path_points)
    feasible = total - len(failed_indices)

    return to_python({
        "type": "PPP",
        "is_feasible": bool(len(failed_indices) == 0),
        "total_points": int(total),
        "feasible_points": int(feasible),
        "failed_points": int(len(failed_indices)),
        "failed_indices": [int(i) for i in failed_indices],
        "feasibility_ratio": float(round(feasible / total, 4)),
        "joint_sequences": {"d1": joint_seq[0], "d2": joint_seq[1], "d3": joint_seq[2]},
        "path_points": [[float(p[0]), float(p[1]), float(p[2])] for p in path_points],
        "per_point_results": results,
    })


def compute_workspace_boundary(link_lengths, num_samples=200):
    from kinematics.forward import forward_kinematics_RRR
    points = []
    angles = np.linspace(-np.pi, np.pi, num_samples // 10)
    for t1 in angles:
        for t2 in angles:
            for t3 in np.linspace(-np.pi, np.pi, 5):
                fk = forward_kinematics_RRR([t1, t2, t3], link_lengths)
                ee = fk["end_effector"]
                points.append([round(float(ee["x"]), 4), round(float(ee["y"]), 4)])

    return to_python({
        "max_reach": float(sum(link_lengths)),
        "min_reach": float(max(0, abs(link_lengths[0] - link_lengths[1]) - link_lengths[2])),
        "workspace_points": points[:2000],
        "link_lengths": link_lengths,
    })


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
