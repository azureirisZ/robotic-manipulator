"""
Inverse Kinematics for RRR and PPP robotic manipulators.
Analytical solution for planar RRR; direct solution for PPP.
Numerical fallback via SciPy for degenerate cases.
"""

import numpy as np
from scipy.optimize import minimize
from .forward import fk_RRR, fk_PPP


def ik_RRR(target_position, link_lengths, elbow="up"):
    """
    Analytical IK for a 3-DOF planar RRR manipulator.

    The third joint is set so the end-effector reaches the exact (x, y)
    target with a chosen orientation.  We solve for theta1, theta2, theta3
    using the geometric (cosine-rule) approach.

    Args:
        target_position : [x, y, phi] where phi is desired end-effector
                          orientation in the XY plane (radians).
                          If only [x, y] is given, phi defaults to 0.
        link_lengths    : [L1, L2, L3]
        elbow           : "up" or "down"

    Returns:
        dict with:
            joint_angles : [theta1, theta2, theta3] or None if unreachable
            reachable    : bool
            message      : status string
    """
    L1, L2, L3 = link_lengths

    if len(target_position) == 3:
        x, y, phi = target_position
    else:
        x, y = target_position[:2]
        phi = 0.0

    # Wrist centre (remove L3 contribution along desired orientation)
    wx = x - L3 * np.cos(phi)
    wy = y - L3 * np.sin(phi)

    r = np.sqrt(wx**2 + wy**2)

    # Reachability check for the 2-link sub-chain (L1, L2)
    if r > L1 + L2 + 1e-9:
        return {"joint_angles": None, "reachable": False,
                "message": "Target out of workspace (too far)"}
    if r < abs(L1 - L2) - 1e-9:
        return {"joint_angles": None, "reachable": False,
                "message": "Target out of workspace (too close)"}

    # Law of cosines → theta2
    cos2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos2 = np.clip(cos2, -1.0, 1.0)
    theta2 = np.arccos(cos2) if elbow == "up" else -np.arccos(cos2)

    # theta1 from geometry
    alpha = np.arctan2(wy, wx)
    beta  = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    theta1 = alpha - beta

    # theta3 so total orientation equals phi
    theta3 = phi - theta1 - theta2

    # Wrap all angles to [-pi, pi]
    angles = [
        np.arctan2(np.sin(t), np.cos(t))
        for t in [theta1, theta2, theta3]
    ]

    # Verify round-trip
    fk = fk_RRR(angles, link_lengths)
    err = np.linalg.norm(np.array(fk["position"][:2]) - np.array([x, y]))
    if err > 1e-6:
        return {"joint_angles": None, "reachable": False,
                "message": f"IK verification failed (err={err:.2e})"}

    return {
        "joint_angles": angles,
        "reachable": True,
        "message": "OK",
    }


def ik_RRR_numerical(target_position, link_lengths, initial_guess=None):
    """
    Numerical IK fallback using SciPy minimisation.
    Minimises position error subject to joint-angle continuity.
    """
    L1, L2, L3 = link_lengths

    if len(target_position) == 3:
        x, y, phi = target_position
    else:
        x, y = target_position[:2]
        phi = 0.0

    if initial_guess is None:
        initial_guess = [0.0, 0.0, 0.0]

    def cost(angles):
        fk = fk_RRR(angles, link_lengths)
        pos_err = np.linalg.norm(np.array(fk["position"][:2]) - np.array([x, y]))**2
        ori_err = (fk["orientation"][2] - phi)**2
        return pos_err + 0.1 * ori_err

    res = minimize(cost, initial_guess, method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 1000})

    if res.fun < 1e-10:
        return {"joint_angles": res.x.tolist(), "reachable": True,
                "message": "Numerical IK converged"}
    return {"joint_angles": None, "reachable": False,
            "message": f"Numerical IK did not converge (cost={res.fun:.2e})"}


def ik_PPP(target_position, axis_directions=None):
    """
    Trivial direct IK for a 3-DOF PPP (Cartesian) manipulator.
    Projects the target onto each axis.

    Args:
        target_position : [x, y, z]
        axis_directions : same as fk_PPP (default X, Y, Z)

    Returns:
        dict with joint_displacements and reachable flag
    """
    if axis_directions is None:
        axis_directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    target = np.array(target_position, dtype=float)
    displacements = [float(np.dot(target, np.array(ax, dtype=float)))
                     for ax in axis_directions]

    return {
        "joint_displacements": displacements,
        "reachable": True,
        "message": "OK",
    }


if __name__ == "__main__":
    print("=== RRR IK round-trip test ===")
    link_lengths = [1.0, 1.0, 0.5]
    target = [1.2, 0.8, 0.3]   # x, y, phi

    result = ik_RRR(target, link_lengths, elbow="up")
    print(f"  IK result  : {result['message']}")
    if result["reachable"]:
        angles = result["joint_angles"]
        print(f"  Joints     : {[round(a, 6) for a in angles]}")
        fk = fk_RRR(angles, link_lengths)
        err = np.linalg.norm(np.array(fk["position"][:2]) - np.array(target[:2]))
        print(f"  FK position: {[round(v, 6) for v in fk['position']]}")
        print(f"  Round-trip err: {err:.2e} m  ✓" if err < 1e-6 else f"  Round-trip err: {err:.2e} m  ✗")

    print("\n=== RRR IK unreachable test ===")
    r2 = ik_RRR([5.0, 5.0], link_lengths)
    print(f"  {r2['message']}")

    print("\n=== PPP IK test ===")
    r3 = ik_PPP([1.0, 2.0, 3.0])
    print(f"  Displacements: {r3['joint_displacements']}")
