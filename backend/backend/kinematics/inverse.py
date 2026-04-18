"""
Inverse Kinematics for 3-DOF Robotic Manipulators
Supports: RRR (Revolute-Revolute-Revolute) and PPP (Prismatic-Prismatic-Prismatic)

RRR: Analytical geometric solution (fast, exact, handles multiple solutions)
PPP: Trivial direct mapping (displacements = target coordinates)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import minimize


@dataclass
class IKResult:
    """Result of an inverse kinematics computation."""
    success: bool
    joint_values: list[float]        # joint angles (RRR) or displacements (PPP)
    joint_values_deg: list[float]    # degrees for RRR, same as above for PPP
    error: float                     # residual position error (meters)
    configuration: str               # "elbow_up", "elbow_down", or "direct"
    message: str


def inverse_kinematics_RRR(
    target_x: float,
    target_y: float,
    target_z: float,
    link_lengths: list[float],
    elbow_up: bool = True,
    joint_limits: list[tuple] = None
) -> IKResult:
    """
    Analytical inverse kinematics for a 3-DOF planar RRR manipulator.

    Uses geometric approach:
      - theta1: base rotation from atan2 of target position
      - theta2, theta3: solved via law of cosines in the plane

    For a planar arm (all joints rotate about Z-axis with alpha=0):
      x = (a1*c1 + a2*c12 + a3*c123)
      y = (a1*s1 + a2*s12 + a3*s123)

    Args:
        target_x, target_y, target_z: desired end-effector position
        link_lengths: [a1, a2, a3]
        elbow_up: True = elbow-up config, False = elbow-down
        joint_limits: [(min1, max1), (min2, max2), (min3, max3)] in radians

    Returns:
        IKResult with joint angles in radians
    """
    a1, a2, a3 = link_lengths

    # Default joint limits: full rotation allowed
    if joint_limits is None:
        joint_limits = [(-np.pi, np.pi)] * 3

    # ── Planar 3-DOF RRR analytical IK ──────────────────────────────────────
    # All joints revolve about Z. The arm lies in the XY plane (z ignored).
    # End-effector position:
    #   px = a1*c1 + a2*c12 + a3*c123
    #   py = a1*s1 + a2*s12 + a3*s123
    # where c1=cos(t1), s12=sin(t1+t2), c123=cos(t1+t2+t3), etc.
    #
    # Strategy (free end-effector orientation):
    #   1. Choose phi = total arm angle (t1+t2+t3) freely → we set phi = atan2(py, px)
    #      so the last link points straight at the target from the wrist.
    #   2. Wrist center = target − a3 * [cos(phi), sin(phi)]
    #   3. Solve 2-link (a1, a2) IK to reach wrist center → gives t1, t2
    #   4. t3 = phi − t1 − t2

    px, py = target_x, target_y

    # Step 1 — total arm angle (end-effector pointing toward target)
    phi = np.arctan2(py, px)

    # Step 2 — wrist center
    wx = px - a3 * np.cos(phi)
    wy = py - a3 * np.sin(phi)
    D  = np.sqrt(wx**2 + wy**2)   # shoulder-to-wrist distance

    # Reachability check for 2-link sub-problem
    max_reach_2 = a1 + a2
    min_reach_2 = abs(a1 - a2)

    if D > max_reach_2:
        return IKResult(
            success=False,
            joint_values=[],
            joint_values_deg=[],
            error=D - max_reach_2,
            configuration="unreachable",
            message=f"Target out of reach. Wrist distance {D:.4f} > a1+a2={max_reach_2:.4f}"
        )
    if D < min_reach_2:
        return IKResult(
            success=False,
            joint_values=[],
            joint_values_deg=[],
            error=min_reach_2 - D,
            configuration="unreachable",
            message=f"Target too close. Wrist distance {D:.4f} < |a1-a2|={min_reach_2:.4f}"
        )

    # Step 3 — elbow angle via law of cosines
    cos_t2 = (D**2 - a1**2 - a2**2) / (2.0 * a1 * a2)
    cos_t2 = np.clip(cos_t2, -1.0, 1.0)

    if elbow_up:
        t2 = np.arccos(cos_t2)
        config_label = "elbow_up"
    else:
        t2 = -np.arccos(cos_t2)
        config_label = "elbow_down"

    # Shoulder angle
    gamma = np.arctan2(wy, wx)
    delta = np.arctan2(a2 * np.sin(t2), a1 + a2 * np.cos(t2))
    t1 = gamma - delta

    # Step 4 — wrist angle closes the chain
    t3 = phi - t1 - t2

    joint_angles = [t1, t2, t3]

    # Check joint limits
    for i, (angle, (lo, hi)) in enumerate(zip(joint_angles, joint_limits)):
        if not (lo <= angle <= hi):
            return IKResult(
                success=False,
                joint_values=joint_angles,
                joint_values_deg=[float(np.degrees(a)) for a in joint_angles],
                error=0.0,
                configuration=config_label,
                message=f"Joint {i + 1} angle {np.degrees(angle):.2f}° exceeds limits [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°]"
            )

    # Verify by running FK and computing position error
    from kinematics.forward import forward_kinematics_RRR
    fk = forward_kinematics_RRR(joint_angles, link_lengths)
    ee = fk["end_effector"]
    error = float(np.sqrt(
        (ee["x"] - target_x)**2 +
        (ee["y"] - target_y)**2
        # z is ignored for planar arm (target_z not used in planar IK)
    ))

    return IKResult(
        success=True,
        joint_values=[float(v) for v in joint_angles],
        joint_values_deg=[float(np.degrees(a)) for a in joint_angles],
        error=error,
        configuration=config_label,
        message=f"Solution found ({config_label}). Position error: {error:.6f} m"
    )


def inverse_kinematics_RRR_numerical(
    target_x: float,
    target_y: float,
    target_z: float,
    link_lengths: list[float],
    initial_guess: list[float] = None,
    joint_limits: list[tuple] = None
) -> IKResult:
    """
    Numerical IK for RRR using scipy optimization.
    Fallback when analytical solution is marginal or for non-standard configs.

    Minimizes position error via L-BFGS-B with joint limit constraints.
    """
    from kinematics.forward import forward_kinematics_RRR

    if initial_guess is None:
        initial_guess = [0.0, np.pi / 4, -np.pi / 4]

    if joint_limits is None:
        joint_limits = [(-np.pi, np.pi)] * 3

    target = np.array([target_x, target_y, target_z])

    def objective(angles):
        try:
            fk = forward_kinematics_RRR(angles.tolist(), link_lengths)
            ee = fk["end_effector"]
            pos = np.array([ee["x"], ee["y"], ee["z"]])
            return float(np.sum((pos - target) ** 2))
        except Exception:
            return 1e6

    result = minimize(
        objective,
        x0=np.array(initial_guess),
        method="L-BFGS-B",
        bounds=joint_limits,
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 1000}
    )

    error = np.sqrt(result.fun) if result.fun > 0 else 0.0
    angles = result.x.tolist()

    return IKResult(
        success=result.success and error < 1e-3,
        joint_values=angles,
        joint_values_deg=[float(np.degrees(a)) for a in angles],
        error=float(error),
        configuration="numerical",
        message=result.message if not result.success else f"Numerical solution found. Error: {error:.6f} m"
    )


def inverse_kinematics_PPP(
    target_x: float,
    target_y: float,
    target_z: float,
    joint_limits: list[tuple] = None
) -> IKResult:
    """
    Inverse kinematics for a 3-DOF PPP (Cartesian) manipulator.

    Trivially direct: each prismatic joint maps to one Cartesian axis.
    d1 = x,  d2 = y,  d3 = z

    Args:
        target_x, target_y, target_z: desired end-effector position
        joint_limits: [(min1, max1), (min2, max2), (min3, max3)] in meters

    Returns:
        IKResult with prismatic displacements
    """
    displacements = [target_x, target_y, target_z]

    # Default limits: 0 to 2 meters per axis
    if joint_limits is None:
        joint_limits = [(0.0, 2.0)] * 3

    for i, (disp, (lo, hi)) in enumerate(zip(displacements, joint_limits)):
        if not (lo <= disp <= hi):
            return IKResult(
                success=False,
                joint_values=displacements,
                joint_values_deg=displacements,
                error=0.0,
                configuration="direct",
                message=f"Joint {i + 1} displacement {disp:.4f} m exceeds limits [{lo}, {hi}] m"
            )

    return IKResult(
        success=True,
        joint_values=displacements,
        joint_values_deg=displacements,   # same units for PPP
        error=0.0,
        configuration="direct",
        message="Direct mapping: displacements = target coordinates"
    )


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    link_lengths = [1.0, 0.8, 0.5]

    print("=== RRR Inverse Kinematics (Elbow Up) ===")
    result = inverse_kinematics_RRR(
        target_x=1.2, target_y=0.5, target_z=0.0,
        link_lengths=link_lengths,
        elbow_up=True
    )
    print(f"Success: {result.success}")
    print(f"Joint angles (deg): {result.joint_values_deg}")
    print(f"FK error: {result.error:.6f} m")
    print(f"Message: {result.message}")

    print("\n=== RRR Inverse Kinematics (Elbow Down) ===")
    result2 = inverse_kinematics_RRR(
        target_x=1.2, target_y=0.5, target_z=0.0,
        link_lengths=link_lengths,
        elbow_up=False
    )
    print(f"Joint angles (deg): {result2.joint_values_deg}")

    print("\n=== RRR IK (Out of Reach) ===")
    result3 = inverse_kinematics_RRR(
        target_x=5.0, target_y=5.0, target_z=0.0,
        link_lengths=link_lengths
    )
    print(f"Success: {result3.success} | {result3.message}")

    print("\n=== PPP Inverse Kinematics ===")
    result_ppp = inverse_kinematics_PPP(
        target_x=0.5, target_y=1.2, target_z=0.8
    )
    print(f"Success: {result_ppp.success}")
    print(f"Displacements: {result_ppp.joint_values}")
