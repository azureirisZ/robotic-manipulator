"""
Forward Kinematics for RRR and PPP robotic manipulators.
Uses DH (Denavit-Hartenberg) convention.
"""

import numpy as np


def dh_transform(a, alpha, d, theta):
    """Compute individual DH transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,       ca,      d],
        [0,        0,        0,      1],
    ])


def fk_RRR(joint_angles, link_lengths):
    """
    Forward kinematics for a 3-DOF planar RRR manipulator.

    Args:
        joint_angles : [theta1, theta2, theta3] in radians
        link_lengths : [L1, L2, L3] in metres

    Returns:
        dict with:
            position      : [x, y, z] of end-effector
            orientation   : [roll, pitch, yaw] in radians
            joint_origins : list of [x, y, z] for base + each joint
            T_final       : 4x4 homogeneous transform
    """
    t1, t2, t3 = joint_angles
    L1, L2, L3 = link_lengths

    # DH params: (a, alpha, d, theta)
    dh_params = [
        (L1, 0, 0, t1),
        (L2, 0, 0, t2),
        (L3, 0, 0, t3),
    ]

    T = np.eye(4)
    origins = [T[:3, 3].copy()]   # base origin

    for a, alpha, d, theta in dh_params:
        T = T @ dh_transform(a, alpha, d, theta)
        origins.append(T[:3, 3].copy())

    R = T[:3, :3]
    roll  = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw   = np.arctan2(R[1, 0], R[0, 0])

    return {
        "position":      T[:3, 3].tolist(),
        "orientation":   [roll, pitch, yaw],
        "joint_origins": [o.tolist() for o in origins],
        "T_final":       T,
    }


def fk_PPP(joint_displacements, axis_directions=None):
    """
    Forward kinematics for a 3-DOF PPP (Cartesian) manipulator.

    Args:
        joint_displacements : [d1, d2, d3] linear displacements in metres
        axis_directions     : list of 3-element unit vectors (default: X, Y, Z)

    Returns:
        dict with position, orientation (always zeros), joint_origins, T_final
    """
    if axis_directions is None:
        axis_directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    T = np.eye(4)
    origins = [np.zeros(3)]

    for d, axis in zip(joint_displacements, axis_directions):
        ax = np.array(axis, dtype=float)
        T_step = np.eye(4)
        T_step[:3, 3] = d * ax
        T = T @ T_step
        origins.append(T[:3, 3].copy())

    return {
        "position":      T[:3, 3].tolist(),
        "orientation":   [0.0, 0.0, 0.0],
        "joint_origins": [o.tolist() for o in origins],
        "T_final":       T,
    }


if __name__ == "__main__":
    print("=== RRR FK test ===")
    angles = [np.pi / 4, np.pi / 4, np.pi / 4]
    lengths = [1.0, 1.0, 0.5]
    result = fk_RRR(angles, lengths)
    print(f"  Position   : {[round(v, 6) for v in result['position']]}")
    print(f"  Orientation: {[round(v, 6) for v in result['orientation']]}")

    print("\n=== PPP FK test ===")
    disp = [1.0, 2.0, 3.0]
    result2 = fk_PPP(disp)
    print(f"  Position: {result2['position']}")
