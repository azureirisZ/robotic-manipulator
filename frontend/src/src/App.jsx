import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Simulator from "./pages/Simulator";
import TrajectoryMode from "./pages/TrajectoryMode";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <nav className="navbar">
          <div className="nav-brand">
            <span className="nav-logo">⚙️</span>
            <span className="nav-title">RoboSim</span>
          </div>
          <div className="nav-links">
            <NavLink to="/" end className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
              Simulator
            </NavLink>
            <NavLink to="/trajectory" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
              Trajectory
            </NavLink>
          </div>
        </nav>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Simulator />} />
            <Route path="/trajectory" element={<TrajectoryMode />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
