"""
Pydantic models for FastAPI request and response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


# ── Shared ────────────────────────────────────────────────────────────────

class LinkLengths(BaseModel):
    a1: float = Field(..., gt=0, description="Link 1 length (m)")
    a2: float = Field(..., gt=0, description="Link 2 length (m)")
    a3: float = Field(..., gt=0, description="Link 3 length (m)")

    def to_list(self):
        return [self.a1, self.a2, self.a3]


class JointLimits(BaseModel):
    min1: float = -3.14159
    max1: float =  3.14159
    min2: float = -3.14159
    max2: float =  3.14159
    min3: float = -3.14159
    max3: float =  3.14159

    def to_list(self):
        return [(self.min1, self.max1), (self.min2, self.max2), (self.min3, self.max3)]


# ── Forward Kinematics ────────────────────────────────────────────────────

class FKRequest(BaseModel):
    robot_type: Literal["RRR", "PPP"] = "RRR"
    joint_values: list[float] = Field(..., min_length=3, max_length=3)
    link_lengths: Optional[list[float]] = None   # required for RRR


class FKResponse(BaseModel):
    type: str
    end_effector: dict
    joint_positions: list
    transform_matrix: list


# ── Inverse Kinematics ────────────────────────────────────────────────────

class IKRequest(BaseModel):
    robot_type: Literal["RRR", "PPP"] = "RRR"
    target_x: float
    target_y: float
    target_z: float = 0.0
    link_lengths: Optional[list[float]] = None   # required for RRR
    elbow_up: bool = True
    use_numerical_fallback: bool = True
    joint_limits: Optional[JointLimits] = None


class IKResponse(BaseModel):
    success: bool
    joint_values: list
    joint_values_deg: list
    error: float
    configuration: str
    message: str


# ── Trajectory ────────────────────────────────────────────────────────────

class PathPoint(BaseModel):
    x: float
    y: float
    z: float = 0.0


class TrajectoryRequest(BaseModel):
    robot_type: Literal["RRR", "PPP"] = "RRR"
    path_type: Literal["custom", "line", "circle"] = "custom"

    # For custom paths
    path_points: Optional[list[PathPoint]] = None

    # For line
    start: Optional[PathPoint] = None
    end: Optional[PathPoint] = None
    num_points: int = Field(50, ge=2, le=500)

    # For circle
    center: Optional[PathPoint] = None
    radius: Optional[float] = None
    plane: Literal["xy", "xz", "yz"] = "xy"

    # Robot config
    link_lengths: Optional[list[float]] = None
    elbow_up: bool = True
    joint_limits: Optional[JointLimits] = None


class TrajectoryResponse(BaseModel):
    type: str
    is_feasible: bool
    total_points: int
    feasible_points: int
    failed_points: int
    failed_indices: list
    feasibility_ratio: float
    joint_sequences: dict
    path_points: list
    link_lengths: Optional[list] = None


# ── Optimizer ─────────────────────────────────────────────────────────────

class OptimizerRequest(BaseModel):
    path_points: list[PathPoint]
    initial_link_lengths: list[float] = Field(..., min_length=3, max_length=3)
    max_total_length: Optional[float] = None
    length_bounds: Optional[list[list[float]]] = None  # [[min,max], ...]
    method: Literal["auto", "local", "global"] = "auto"
    joint_limits: Optional[JointLimits] = None


class OptimizerResponse(BaseModel):
    success: bool
    already_feasible: bool
    optimized_link_lengths: list
    initial_link_lengths: list
    length_changes: list
    total_change: float
    feasibility_ratio: float
    message: str
