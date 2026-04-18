"""
FastAPI backend for the Robotic Manipulator Simulator.

Endpoints:
  POST /fk           — Forward kinematics
  POST /ik           — Inverse kinematics
  POST /trajectory   — Trajectory tracking + feasibility
  POST /optimize     — Link length optimization
  GET  /workspace    — Workspace boundary point cloud
  GET  /health       — Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    FKRequest, FKResponse,
    IKRequest, IKResponse,
    TrajectoryRequest, TrajectoryResponse,
    OptimizerRequest, OptimizerResponse,
)
from kinematics.forward import forward_kinematics_RRR, forward_kinematics_PPP
from kinematics.inverse import (
    inverse_kinematics_RRR,
    inverse_kinematics_RRR_numerical,
    inverse_kinematics_PPP,
)
from kinematics.trajectory import (
    track_trajectory_RRR,
    track_trajectory_PPP,
    generate_line_trajectory,
    generate_circle_trajectory,
    compute_workspace_boundary,
)
from kinematics.optimizer import optimize_link_lengths

app = FastAPI(
    title="Robotic Manipulator API",
    description="Simulation and analysis of 3-DOF RRR and PPP robotic manipulators",
    version="1.0.0",
)

# Allow all origins for development — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Forward Kinematics ────────────────────────────────────────────────────

@app.post("/fk")
def forward_kinematics(req: FKRequest):
    """
    Compute end-effector position and orientation from joint variables.

    - RRR: joint_values = [theta1, theta2, theta3] in radians
    - PPP: joint_values = [d1, d2, d3] in meters
    """
    try:
        if req.robot_type == "RRR":
            if not req.link_lengths or len(req.link_lengths) != 3:
                raise HTTPException(status_code=422, detail="RRR requires link_lengths [a1, a2, a3]")
            result = forward_kinematics_RRR(req.joint_values, req.link_lengths)
        else:
            result = forward_kinematics_PPP(req.joint_values)

        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Inverse Kinematics ────────────────────────────────────────────────────

@app.post("/ik")
def inverse_kinematics(req: IKRequest):
    """
    Compute joint variables from a desired end-effector position.

    Returns both elbow-up and elbow-down solutions for RRR.
    """
    try:
        jlimits = req.joint_limits.to_list() if req.joint_limits else None

        if req.robot_type == "PPP":
            ik = inverse_kinematics_PPP(req.target_x, req.target_y, req.target_z, jlimits)
            return {
                "success": ik.success,
                "joint_values": ik.joint_values,
                "joint_values_deg": ik.joint_values_deg,
                "error": ik.error,
                "configuration": ik.configuration,
                "message": ik.message,
            }

        # RRR
        if not req.link_lengths or len(req.link_lengths) != 3:
            raise HTTPException(status_code=422, detail="RRR requires link_lengths [a1, a2, a3]")

        ik = inverse_kinematics_RRR(
            req.target_x, req.target_y, req.target_z,
            req.link_lengths, req.elbow_up, jlimits
        )

        # Numerical fallback
        if not ik.success and req.use_numerical_fallback:
            ik = inverse_kinematics_RRR_numerical(
                req.target_x, req.target_y, req.target_z,
                req.link_lengths, joint_limits=jlimits
            )

        # Also compute the alternate elbow config for the response
        ik_alt = inverse_kinematics_RRR(
            req.target_x, req.target_y, req.target_z,
            req.link_lengths, not req.elbow_up, jlimits
        )

        return {
            "success": ik.success,
            "joint_values": ik.joint_values,
            "joint_values_deg": ik.joint_values_deg,
            "error": ik.error,
            "configuration": ik.configuration,
            "message": ik.message,
            "alternate_solution": {
                "success": ik_alt.success,
                "joint_values": ik_alt.joint_values,
                "joint_values_deg": ik_alt.joint_values_deg,
                "configuration": ik_alt.configuration,
            } if ik_alt.success else None,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Trajectory Tracking ───────────────────────────────────────────────────

@app.post("/trajectory")
def trajectory(req: TrajectoryRequest):
    """
    Track a trajectory and check workspace feasibility.

    Supports custom waypoints, line segments, and circular paths.
    Returns per-point IK results and joint variable sequences.
    """
    try:
        # Build path points
        if req.path_type == "line":
            if not req.start or not req.end:
                raise HTTPException(status_code=422, detail="Line trajectory requires start and end")
            path = generate_line_trajectory(
                [req.start.x, req.start.y, req.start.z],
                [req.end.x,   req.end.y,   req.end.z],
                req.num_points,
            )
        elif req.path_type == "circle":
            if not req.center or not req.radius:
                raise HTTPException(status_code=422, detail="Circle trajectory requires center and radius")
            path = generate_circle_trajectory(
                [req.center.x, req.center.y, req.center.z],
                req.radius, req.num_points, req.plane,
            )
        else:  # custom
            if not req.path_points:
                raise HTTPException(status_code=422, detail="Custom trajectory requires path_points")
            path = [[p.x, p.y, p.z] for p in req.path_points]

        jlimits = req.joint_limits.to_list() if req.joint_limits else None

        if req.robot_type == "PPP":
            return track_trajectory_PPP(path, jlimits)

        if not req.link_lengths or len(req.link_lengths) != 3:
            raise HTTPException(status_code=422, detail="RRR requires link_lengths [a1, a2, a3]")

        return track_trajectory_RRR(path, req.link_lengths, jlimits, req.elbow_up)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Link Length Optimizer ─────────────────────────────────────────────────

@app.post("/optimize")
def optimize(req: OptimizerRequest):
    """
    Find minimum link length adjustments to make an infeasible trajectory reachable.

    Uses SciPy local + global optimization (differential evolution).
    """
    try:
        path = [[p.x, p.y, p.z] for p in req.path_points]
        jlimits = req.joint_limits.to_list() if req.joint_limits else None
        bounds = [tuple(b) for b in req.length_bounds] if req.length_bounds else None

        return optimize_link_lengths(
            path,
            req.initial_link_lengths,
            joint_limits=jlimits,
            max_total_length=req.max_total_length,
            length_bounds=bounds,
            method=req.method,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Workspace ─────────────────────────────────────────────────────────────

@app.get("/workspace")
def workspace(a1: float = 1.0, a2: float = 0.8, a3: float = 0.5):
    """
    Return workspace boundary data for an RRR manipulator.
    Query params: a1, a2, a3 (link lengths in meters).
    """
    try:
        return compute_workspace_boundary([a1, a2, a3])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
