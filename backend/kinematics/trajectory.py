import numpy as np
from kinematics.inverse import inverse_kinematics_RRR

def track_trajectory(points, link_lengths, elbow_up=True):
    joint_trajectory = []
    errors = []
    
    for point in points:
        x, y = point

        result = inverse_kinematics_RRR(
            target_x=x,
            target_y=y,
            target_z=0.0,
            link_lengths=link_lengths,
            elbow_up=elbow_up
        )

        if not result.success:
            return {
                "success": False,
                "failed_point": point,
                "message": result.message
            }

        joint_trajectory.append(result.joint_values)
        errors.append(result.error)

    return {
        "success": True,
        "joint_trajectory": joint_trajectory,
        "avg_error": float(np.mean(errors))
    }
