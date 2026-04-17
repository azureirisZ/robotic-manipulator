import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000"
});

export const getIK = (data) => API.post("/ik", data);
export const getFK = (data) => API.post("/fk", data);
export const getTrajectory = (data) => API.post("/trajectory", data);
