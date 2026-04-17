"""
Link length optimizer for RRR manipulators.

When a trajectory is infeasible with the current link lengths,
this module uses SciPy to find the minimum-change link lengths
that make the full trajectory reachable.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from .trajectory import track_trajectory_RRR


def optimize_link_lengths(
    path_points,
    current_link_lengths,
    elbow="up",
    phi=0.0,
    min_length=0.1,
    max_length=5.0,
    method="scipy",
):
    """
    Find link lengths closest to `current_link_lengths` that make the
    full trajectory feasible.

    Args:
        path_points          : list of [x, y] or [x, y, phi] trajectory points
        current_link_lengths : [L1, L2, L3] — starting point for optimisation
        elbow                : "up" or "down"
        phi                  : default end-effector orientation
        min_length           : lower bound for each link (metres)
        max_length           : upper bound for each link (metres)
        method               : "scipy" (local) or "global" (differential evolution)

    Returns:
        dict with:
            success              : bool
            optimized_lengths    : [L1, L2, L3]
            delta                : change from original lengths [dL1, dL2, dL3]
            feasibility_fraction : fraction of trajectory reachable with new lengths
            message              : status string
    """
    L0 = np.array(current_link_lengths, dtype=float)

    def infeasibility_cost(L):
        """Penalise: fraction unreachable + deviation from original lengths."""
        L = np.clip(L, min_length, max_length)
        result = track_trajectory_RRR(path_points, L.tolist(), elbow=elbow, phi=phi)
        n_total = len(path_points)
        n_bad   = len(result["infeasible_pts"])
        infeasibility_penalty = n_bad / n_total          # 0 → perfect, 1 → all bad
        length_penalty = np.linalg.norm(L - L0) / np.linalg.norm(L0)  # relative change
        return infeasibility_penalty * 10.0 + length_penalty

    bounds = [(min_length, max_length)] * 3

    if method == "global":
        # Slower but avoids local minima — good for very infeasible trajectories
        result = differential_evolution(
            infeasibility_cost,
            bounds,
            seed=42,
            maxiter=200,
            tol=1e-4,
            popsize=10,
        )
        opt_L = result.x
    else:
        # Fast local search from current lengths
        result = minimize(
            infeasibility_cost,
            L0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-9, "maxiter": 500},
        )
        opt_L = result.x

    opt_L = np.clip(opt_L, min_length, max_length).tolist()

    # Evaluate the optimised result
    final = track_trajectory_RRR(path_points, opt_L, elbow=elbow, phi=phi)
    frac  = sum(final["reachable_mask"]) / len(path_points)

    success = final["feasible"]
    delta   = (np.array(opt_L) - L0).tolist()

    if success:
        msg = (
            f"Optimisation successful. "
            f"New lengths: {[round(l, 4) for l in opt_L]}  "
            f"(Δ = {[round(d, 4) for d in delta]})"
        )
    else:
        msg = (
            f"Optimisation improved feasibility to {frac*100:.1f}% "
            f"but could not reach 100%. "
            f"Suggested lengths: {[round(l, 4) for l in opt_L]}"
        )

    return {
        "success":               success,
        "optimized_lengths":     opt_L,
        "delta":                 delta,
        "feasibility_fraction":  frac,
        "message":               msg,
    }


if __name__ == "__main__":
    from .trajectory import line_path, circle_path

    print("=== Optimizer: fix an unreachable line trajectory ===")
    # Path that's too far for [1, 1, 0.5]
    bad_path = line_path([2.5, 0.0], [3.0, 0.0], n_points=15)
    result = optimize_link_lengths(bad_path, [1.0, 1.0, 0.5])
    print(f"  Success        : {result['success']}")
    print(f"  Opt lengths    : {[round(l, 4) for l in result['optimized_lengths']]}")
    print(f"  Delta          : {[round(d, 4) for d in result['delta']]}")
    print(f"  Feasibility    : {result['feasibility_fraction']*100:.1f}%")
    print(f"  {result['message']}")

    print("\n=== Optimizer: already-feasible trajectory (should barely change) ===")
    good_path = line_path([0.8, 0.2], [1.0, 0.5], n_points=10)
    result2 = optimize_link_lengths(good_path, [1.0, 1.0, 0.5])
    print(f"  Success     : {result2['success']}")
    print(f"  Opt lengths : {[round(l, 4) for l in result2['optimized_lengths']]}")
    print(f"  {result2['message']}")
