"""
Link Length Optimizer for 3-DOF RRR Manipulator

When a trajectory is partially or fully outside the workspace,
this module finds the minimum adjustment to link lengths that
makes the entire trajectory reachable.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from kinematics.trajectory import track_trajectory_RRR


def optimize_link_lengths(
    path_points: list,
    initial_link_lengths: list,
    joint_limits: list = None,
    max_total_length: float = None,
    length_bounds: list = None,
    method: str = "auto",
) -> dict:
    """
    Find minimum-change link lengths that make a trajectory fully reachable.

    Args:
        path_points:           list of [x, y, z] target positions
        initial_link_lengths:  [a1, a2, a3] starting configuration
        joint_limits:          joint angle limits (passed to trajectory tracker)
        max_total_length:      optional cap on a1+a2+a3
        length_bounds:         [(min, max)] per link, default [(0.1, 5.0)] * 3
        method:                "local" (fast), "global" (thorough), or "auto"

    Returns:
        dict with optimized link lengths, improvement metrics, and convergence info
    """
    a0 = np.array(initial_link_lengths, dtype=float)

    if length_bounds is None:
        length_bounds = [(0.1, 5.0)] * 3

    if max_total_length is None:
        max_total_length = sum(initial_link_lengths) * 2.5

    # First check if trajectory is already feasible
    initial_result = track_trajectory_RRR(path_points, initial_link_lengths, joint_limits)
    if initial_result["is_feasible"]:
        return {
            "success": True,
            "already_feasible": True,
            "optimized_link_lengths": initial_link_lengths,
            "initial_link_lengths": initial_link_lengths,
            "length_changes": [0.0, 0.0, 0.0],
            "total_change": 0.0,
            "feasibility_ratio": 1.0,
            "message": "Trajectory already feasible with current link lengths.",
        }

    def objective(links):
        """Minimize: infeasible fraction + small penalty for large changes."""
        links = np.abs(links)  # links must be positive
        result = track_trajectory_RRR(path_points, links.tolist(), joint_limits)
        infeasible_ratio = 1.0 - result["feasibility_ratio"]

        # Penalty: deviation from initial lengths (normalized)
        change_penalty = 0.05 * np.sum(((links - a0) / a0) ** 2)

        # Total length constraint penalty
        total_penalty = 0.0
        if np.sum(links) > max_total_length:
            total_penalty = 10.0 * (np.sum(links) - max_total_length)

        return infeasible_ratio + change_penalty + total_penalty

    # Auto: try local first, upgrade to global if needed
    if method == "auto":
        result_local = minimize(
            objective,
            x0=a0 * 1.2,   # start slightly larger
            method="L-BFGS-B",
            bounds=length_bounds,
            options={"ftol": 1e-8, "maxiter": 500},
        )
        best = result_local

        # Check if local found a fully feasible solution
        check = track_trajectory_RRR(path_points, np.abs(best.x).tolist(), joint_limits)
        if not check["is_feasible"]:
            # Escalate to global search
            result_global = differential_evolution(
                objective,
                bounds=length_bounds,
                maxiter=300,
                tol=1e-6,
                seed=42,
                workers=1,
                polish=True,
            )
            if result_global.fun < best.fun:
                best = result_global

    elif method == "global":
        best = differential_evolution(
            objective,
            bounds=length_bounds,
            maxiter=500,
            tol=1e-7,
            seed=42,
            workers=1,
            polish=True,
        )
    else:  # local
        best = minimize(
            objective,
            x0=a0 * 1.2,
            method="L-BFGS-B",
            bounds=length_bounds,
            options={"ftol": 1e-8, "maxiter": 500},
        )

    optimized = np.abs(best.x).tolist()
    optimized = [round(v, 4) for v in optimized]

    # Final verification
    final_result = track_trajectory_RRR(path_points, optimized, joint_limits)
    changes = [round(optimized[i] - initial_link_lengths[i], 4) for i in range(3)]
    total_change = round(sum(abs(c) for c in changes), 4)

    return {
        "success": final_result["is_feasible"],
        "already_feasible": False,
        "optimized_link_lengths": optimized,
        "initial_link_lengths": initial_link_lengths,
        "length_changes": changes,
        "total_change": total_change,
        "feasibility_ratio": final_result["feasibility_ratio"],
        "feasible_points": final_result["feasible_points"],
        "total_points": final_result["total_points"],
        "optimizer_value": float(best.fun),
        "message": (
            f"Optimized successfully. Total link length change: {total_change:.4f} m"
            if final_result["is_feasible"]
            else f"Partial improvement: {final_result['feasibility_ratio']*100:.1f}% reachable. "
                 "Consider relaxing length bounds or reducing trajectory radius."
        ),
    }


if __name__ == "__main__":
    from kinematics.trajectory import generate_line_trajectory, generate_circle_trajectory

    print("=== Optimizer: out-of-reach line trajectory ===")
    path = generate_line_trajectory([1.8, 0.5, 0.0], [2.0, 1.0, 0.0], num_points=10)
    initial = [1.0, 0.8, 0.5]

    result = optimize_link_lengths(path, initial)
    print(f"Success: {result['success']}")
    print(f"Initial links: {result['initial_link_lengths']}")
    print(f"Optimized links: {result['optimized_link_lengths']}")
    print(f"Changes: {result['length_changes']}")
    print(f"Total change: {result['total_change']} m")
    print(f"Feasibility: {result['feasibility_ratio'] * 100:.1f}%")
    print(f"Message: {result['message']}")
