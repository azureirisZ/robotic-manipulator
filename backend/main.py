from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from kinematics.forward import forward_kinematics_RRR
from kinematics.inverse import inverse_kinematics_RRR
from kinematics.trajectory import track_trajectory

app = FastAPI()

class FKRequest(BaseModel):
    joint_angles: List[float]
    link_lengths: List[float]

class IKRequest(BaseModel):
    x: float
    y: float
    z: float = 0.0
    link_lengths: List[float]
    elbow_up: bool = True

class TrajectoryRequest(BaseModel):
    points: List[List[float]]
    link_lengths: List[float]
    elbow_up: bool = True


@app.get("/")
def root():
    return {"message": "Robotic Manipulator API is running 🚀"}


@app.post("/fk")
def compute_fk(req: FKRequest):
    return forward_kinematics_RRR(
        joint_angles=req.joint_angles,
        link_lengths=req.link_lengths
    )


@app.post("/ik")
def compute_ik(req: IKRequest):
    result = inverse_kinematics_RRR(
        target_x=req.x,
        target_y=req.y,
        target_z=req.z,
        link_lengths=req.link_lengths,
        elbow_up=req.elbow_up
    )
    return result.__dict__


@app.post("/trajectory")
def compute_trajectory(req: TrajectoryRequest):
    return track_trajectory(
        points=req.points,
        link_lengths=req.link_lengths,
        elbow_up=req.elbow_up
    )
