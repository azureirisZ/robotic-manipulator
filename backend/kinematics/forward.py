"""
Forward Kinematics for 3-DOF Robotic Manipulators
Supports: RRR (Revolute-Revolute-Revolute) and PPP (Prismatic-Prismatic-Prismatic)

Uses Denavit-Hartenberg (DH) convention to compute end-effector pose.
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class DHParams:
    """Denavit-Hartenberg parameters for a single joint/link."""
    a: float      # link length (distance along x-axis)
    alpha: float  # link twist (rotation about x-axis, radians)
    d: float      # link offset (distance along z-axis)
    theta: float  # joint angle (rotation about z-axis, radians)


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Compute the 4x4 homogeneous DH transformation matrix for a single joint.

    T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct,  -st * ca,   st * sa,  a * ct],
        [st,   ct * ca,  -ct * sa,  a * st],
        [0,    sa,         ca,       d     ],
        [0,    0,          0,        1     ]
    ])


def forward_kinematics_RRR(
    joint_angles: list[float],
    link_lengths: list[float],
    d_offsets: list[float] = None,
    alpha_values: list[float] = None
) -> dict:
    """
    Forward kinematics for a 3-DOF RRR (all-revolute) manipulator.

    Args:
        joint_angles:  [theta1, theta2, theta3] in radians
        link_lengths:  [a1, a2, a3] link lengths in meters
        d_offsets:     [d1, d2, d3] prismatic offsets (default all 0)
        alpha_values:  [alpha1, alpha2, alpha3] twist angles (default all 0 = planar)

    Returns:
        dict with end-effector position, orientation, and full transform matrix
    """
    if len(joint_angles) != 3 or len(link_lengths) != 3:
        raise ValueError("RRR manipulator requires exactly 3 joint angles and 3 link lengths")

    d = d_offsets if d_offsets else [0.0, 0.0, 0.0]
    alpha = alpha_values if alpha_values else [0.0, 0.0, 0.0]
    a = link_lengths
    theta = joint_angles

    # Compute individual DH transforms
    T1 = dh_transform(a[0], alpha[0], d[0], theta[0])
    T2 = dh_transform(a[1], alpha[1], d[1], theta[1])
    T3 = dh_transform(a[2], alpha[2], d[2], theta[2])

    # Full transform: base to end-effector
    T_0_1 = T1
    T_0_2 = T_0_1 @ T2
    T_0_3 = T_0_2 @ T3

    # Extract position and orientation
    position = T_0_3[:3, 3]
    rotation_matrix = T_0_3[:3, :3]

    # Extract Euler angles (ZYX convention)
    roll, pitch, yaw = rotation_matrix_to_euler(rotation_matrix)

    # Collect intermediate joint positions for visualization
    joint_positions = [
        np.array([0.0, 0.0, 0.0]),          # base
        T_0_1[:3, 3],                         # joint 2
        T_0_2[:3, 3],                         # joint 3
        position                              # end-effector
    ]

    return {
        "type": "RRR",
        "end_effector": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        },
        "transform_matrix": T_0_3.tolist(),
        "joint_positions": [p.tolist() for p in joint_positions],
        "joint_angles_deg": [float(np.degrees(t)) for t in theta],
    }


def forward_kinematics_PPP(
    prismatic_displacements: list[float],
    base_position: list[float] = None
) -> dict:
    """
    Forward kinematics for a 3-DOF PPP (all-prismatic) manipulator.

    For a Cartesian (PPP) robot, joint displacements directly map to
    XYZ displacements along the base axes.

    Args:
        prismatic_displacements: [d1, d2, d3] displacements along X, Y, Z axes
        base_position: [x0, y0, z0] base origin offset (default [0, 0, 0])

    Returns:
        dict with end-effector position and full transform matrix
    """
    if len(prismatic_displacements) != 3:
        raise ValueError("PPP manipulator requires exactly 3 prismatic displacements")

    base = base_position if base_position else [0.0, 0.0, 0.0]
    d1, d2, d3 = prismatic_displacements

    # Each joint translates along one axis — DH parameters for Cartesian robot
    # Joint 1: translate along X (alpha=0, a=0, theta=0, d=d1)
    # Joint 2: translate along Y (alpha=90°, a=0, theta=0, d=d2)
    # Joint 3: translate along Z (alpha=-90°, a=0, theta=0, d=d3)
    T1 = dh_transform(0, 0,             d1, 0)
    T2 = dh_transform(0, np.pi / 2,     d2, 0)
    T3 = dh_transform(0, -np.pi / 2,    d3, 0)

    T_0_1 = T1
    T_0_2 = T_0_1 @ T2
    T_0_3 = T_0_2 @ T3

    position = T_0_3[:3, 3] + np.array(base)

    joint_positions = [
        np.array(base),
        (T_0_1[:3, 3] + np.array(base)),
        (T_0_2[:3, 3] + np.array(base)),
        position
    ]

    return {
        "type": "PPP",
        "end_effector": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        },
        "transform_matrix": T_0_3.tolist(),
        "joint_positions": [p.tolist() for p in joint_positions],
        "prismatic_displacements": [float(d1), float(d2), float(d3)],
    }


def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """
    Convert a 3x3 rotation matrix to ZYX Euler angles (roll, pitch, yaw).
    Returns angles in radians.
    """
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))

    if np.abs(np.cos(pitch)) < 1e-6:  # gimbal lock
        roll = 0.0
        yaw = np.arctan2(R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== RRR Forward Kinematics Test ===")
    result = forward_kinematics_RRR(
        joint_angles=[np.pi / 4, np.pi / 3, np.pi / 6],
        link_lengths=[1.0, 0.8, 0.5]
    )
    ee = result["end_effector"]
    print(f"End-effector position: x={ee['x']:.4f}, y={ee['y']:.4f}, z={ee['z']:.4f}")
    print(f"Joint positions: {result['joint_positions']}")

    print("\n=== PPP Forward Kinematics Test ===")
    result_ppp = forward_kinematics_PPP(
        prismatic_displacements=[1.0, 0.5, 0.3]
    )
    ee_ppp = result_ppp["end_effector"]
    print(f"End-effector position: x={ee_ppp['x']:.4f}, y={ee_ppp['y']:.4f}, z={ee_ppp['z']:.4f}")
