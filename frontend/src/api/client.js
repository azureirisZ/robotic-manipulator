import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
});

// Forward Kinematics
export const solveFK = (robotType, jointValues, linkLengths) =>
  api.post("/fk", {
    robot_type: robotType,
    joint_values: jointValues,
    link_lengths: linkLengths,
  });

// Inverse Kinematics
export const solveIK = (robotType, x, y, z, linkLengths, elbowUp = true) =>
  api.post("/ik", {
    robot_type: robotType,
    target_x: x,
    target_y: y,
    target_z: z,
    link_lengths: linkLengths,
    elbow_up: elbowUp,
    use_numerical_fallback: true,
  });

// Trajectory tracking
export const trackTrajectory = (payload) => api.post("/trajectory", payload);

// Link length optimizer
export const optimizeLinks = (pathPoints, initialLinkLengths, method = "auto") =>
  api.post("/optimize", {
    path_points: pathPoints,
    initial_link_lengths: initialLinkLengths,
    method,
  });

// Workspace boundary
export const getWorkspace = (a1, a2, a3) =>
  api.get("/workspace", { params: { a1, a2, a3 } });

export default api;
