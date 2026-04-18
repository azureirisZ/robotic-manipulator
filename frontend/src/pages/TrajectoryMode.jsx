import { useState } from "react";
import { trackTrajectory, optimizeLinks } from "../api/client";
import RobotViewer from "../components/RobotViewer";
import JointChart from "../components/JointChart";

const DEFAULT_LINKS = [1.0, 0.8, 0.5];

export default function TrajectoryMode() {
  const [robotType, setRobotType]       = useState("RRR");
  const [pathType, setPathType]         = useState("line");
  const [links, setLinks]               = useState(DEFAULT_LINKS);
  const [numPoints, setNumPoints]       = useState(50);
  const [elbowUp, setElbowUp]           = useState(true);

  // Line params
  const [lineStart, setLineStart]       = useState({ x: 0.5, y: 0.0, z: 0.0 });
  const [lineEnd, setLineEnd]           = useState({ x: 1.2, y: 0.8, z: 0.0 });

  // Circle params
  const [circCenter, setCircCenter]     = useState({ x: 0.0, y: 0.0, z: 0.0 });
  const [circRadius, setCircRadius]     = useState(1.2);
  const [circPlane, setCircPlane]       = useState("xy");

  const [trajResult, setTrajResult]     = useState(null);
  const [optResult, setOptResult]       = useState(null);
  const [loading, setLoading]           = useState(false);
  const [optLoading, setOptLoading]     = useState(false);
  const [error, setError]               = useState("");

  const updateLink = (i, val) => {
    const updated = [...links];
    updated[i] = parseFloat(val) || 0;
    setLinks(updated);
  };

  const buildPayload = () => {
    const base = {
      robot_type: robotType,
      path_type: pathType,
      num_points: numPoints,
      link_lengths: robotType === "RRR" ? links : null,
      elbow_up: elbowUp,
    };
    if (pathType === "line") {
      return { ...base, start: lineStart, end: lineEnd };
    }
    return { ...base, center: circCenter, radius: circRadius, plane: circPlane };
  };

  const handleTrack = async () => {
    setLoading(true);
    setError("");
    setOptResult(null);
    try {
      const res = await trackTrajectory(buildPayload());
      setTrajResult(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleOptimize = async () => {
    if (!trajResult || !trajResult.path_points) return;
    setOptLoading(true);
    try {
      const pts = trajResult.path_points.map(([x, y, z]) => ({ x, y, z }));
      const res = await optimizeLinks(pts, links);
      setOptResult(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setOptLoading(false);
    }
  };

  const feasPct = trajResult ? (trajResult.feasibility_ratio * 100).toFixed(1) : null;
  const feasColor = trajResult?.is_feasible ? "var(--success)" : trajResult?.feasibility_ratio > 0.5 ? "var(--warning)" : "var(--danger)";

  // Build a fake fkResult for the viewer (last successful point)
  const viewerFk = trajResult?.per_point_results
    ? (() => {
        const last = [...trajResult.per_point_results].reverse().find(p => p.success);
        if (!last) return null;
        // We don't have joint positions per point, so just show path
        return { joint_positions: null, link_lengths: links };
      })()
    : null;

  return (
    <div>
      <div className="page-header">
        <h1>Trajectory Tracking</h1>
        <p>Check workspace feasibility and compute joint variables along a full path</p>
      </div>

      <div className="grid-2">
        {/* ── Left panel ── */}
        <div>
          {/* Robot type */}
          <div className="card" style={{ marginBottom: "1rem" }}>
            <div className="card-title">Robot Type</div>
            <div className="toggle-group">
              {["RRR", "PPP"].map((t) => (
                <button key={t} className={`toggle-btn ${robotType === t ? "active" : ""}`}
                  onClick={() => { setRobotType(t); setTrajResult(null); setOptResult(null); }}>
                  {t}
                </button>
              ))}
            </div>

            {robotType === "RRR" && (
              <>
                <div className="form-group">
                  <label>Link Lengths (m)</label>
                  <div className="form-row">
                    {["a1", "a2", "a3"].map((label, i) => (
                      <div key={i}>
                        <label style={{ fontSize: "0.75rem", color: "var(--accent)" }}>{label}</label>
                        <input type="number" step="0.1" min="0.1"
                          value={links[i]} onChange={(e) => updateLink(i, e.target.value)}/>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="toggle-group">
                  <button className={`toggle-btn ${elbowUp ? "active" : ""}`} onClick={() => setElbowUp(true)}>Elbow Up</button>
                  <button className={`toggle-btn ${!elbowUp ? "active" : ""}`} onClick={() => setElbowUp(false)}>Elbow Down</button>
                </div>
              </>
            )}
          </div>

          {/* Path config */}
          <div className="card" style={{ marginBottom: "1rem" }}>
            <div className="card-title">Path Configuration</div>
            <div className="toggle-group">
              {["line", "circle"].map((t) => (
                <button key={t} className={`toggle-btn ${pathType === t ? "active" : ""}`}
                  onClick={() => setPathType(t)}>
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>

            {pathType === "line" ? (
              <>
                <div className="form-group">
                  <label>Start Point (m)</label>
                  <div className="form-row">
                    {["x", "y", "z"].map((ax) => (
                      <div key={ax}>
                        <label style={{ fontSize: "0.75rem", color: "var(--accent2)" }}>{ax.toUpperCase()}</label>
                        <input type="number" step="0.1" value={lineStart[ax]}
                          onChange={(e) => setLineStart({ ...lineStart, [ax]: parseFloat(e.target.value) || 0 })}/>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="form-group">
                  <label>End Point (m)</label>
                  <div className="form-row">
                    {["x", "y", "z"].map((ax) => (
                      <div key={ax}>
                        <label style={{ fontSize: "0.75rem", color: "var(--accent2)" }}>{ax.toUpperCase()}</label>
                        <input type="number" step="0.1" value={lineEnd[ax]}
                          onChange={(e) => setLineEnd({ ...lineEnd, [ax]: parseFloat(e.target.value) || 0 })}/>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="form-group">
                  <label>Center (m)</label>
                  <div className="form-row">
                    {["x", "y", "z"].map((ax) => (
                      <div key={ax}>
                        <label style={{ fontSize: "0.75rem", color: "var(--accent2)" }}>{ax.toUpperCase()}</label>
                        <input type="number" step="0.1" value={circCenter[ax]}
                          onChange={(e) => setCircCenter({ ...circCenter, [ax]: parseFloat(e.target.value) || 0 })}/>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="form-row-2">
                  <div className="form-group">
                    <label>Radius (m)</label>
                    <input type="number" step="0.1" min="0.1" value={circRadius}
                      onChange={(e) => setCircRadius(parseFloat(e.target.value) || 0.1)}/>
                  </div>
                  <div className="form-group">
                    <label>Plane</label>
                    <select value={circPlane} onChange={(e) => setCircPlane(e.target.value)}>
                      <option value="xy">XY</option>
                      <option value="xz">XZ</option>
                      <option value="yz">YZ</option>
                    </select>
                  </div>
                </div>
              </>
            )}

            <div className="form-group">
              <label>Number of Points: {numPoints}</label>
              <input type="range" min="10" max="200" step="10" value={numPoints}
                onChange={(e) => setNumPoints(parseInt(e.target.value))}
                style={{ width: "100%", accentColor: "var(--accent)" }}/>
            </div>

            <button className="btn btn-primary" onClick={handleTrack} disabled={loading}>
              {loading ? "Computing..." : "Track Trajectory"}
            </button>
            {error && <div className="error-msg">{error}</div>}
          </div>

          {/* Results */}
          {trajResult && (
            <div className="card" style={{ marginBottom: "1rem" }}>
              <div className="card-title">Feasibility Report</div>
              <div className="result-row">
                <span className="result-label">Status</span>
                <span className={`badge ${trajResult.is_feasible ? "badge-success" : "badge-danger"}`}>
                  {trajResult.is_feasible ? "✓ Fully Feasible" : "✗ Partially Infeasible"}
                </span>
              </div>
              <div className="result-row">
                <span className="result-label">Reachable points</span>
                <span className="result-value">{trajResult.feasible_points} / {trajResult.total_points}</span>
              </div>
              <div style={{ margin: "0.5rem 0" }}>
                <div className="feasibility-bar">
                  <div className="feasibility-fill" style={{ width: `${feasPct}%`, background: feasColor }}/>
                </div>
                <div style={{ textAlign: "right", fontSize: "0.82rem", color: feasColor }}>{feasPct}%</div>
              </div>
              {trajResult.failed_indices.length > 0 && (
                <div style={{ marginTop: "0.5rem" }}>
                  <div className="card-title">Failed Point Indices</div>
                  <div className="scroll-list">
                    {trajResult.failed_indices.join(", ")}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Optimizer */}
          {trajResult && !trajResult.is_feasible && robotType === "RRR" && (
            <div className="card">
              <div className="card-title">Link Length Optimizer</div>
              <p style={{ fontSize: "0.85rem", color: "var(--text2)", marginBottom: "1rem" }}>
                Trajectory is not fully reachable. Find minimum link length adjustments to make it feasible.
              </p>
              <button className="btn btn-success" onClick={handleOptimize} disabled={optLoading}>
                {optLoading ? "Optimizing..." : "Optimize Link Lengths"}
              </button>

              {optResult && (
                <div style={{ marginTop: "1rem" }}>
                  <div className="result-row">
                    <span className="result-label">Status</span>
                    <span className={`badge ${optResult.success ? "badge-success" : "badge-warning"}`}>
                      {optResult.success ? "✓ Solution found" : "⚠ Partial improvement"}
                    </span>
                  </div>
                  {["a1", "a2", "a3"].map((label, i) => (
                    <div className="result-row" key={i}>
                      <span className="result-label">{label}</span>
                      <span className="result-value">
                        {optResult.initial_link_lengths[i].toFixed(3)}
                        <span style={{ color: optResult.length_changes[i] > 0 ? "var(--success)" : "var(--text2)", margin: "0 6px" }}>
                          → {optResult.optimized_link_lengths[i].toFixed(3)} m
                        </span>
                        {optResult.length_changes[i] > 0 && (
                          <span style={{ color: "var(--success)", fontSize: "0.78rem" }}>
                            (+{optResult.length_changes[i].toFixed(3)})
                          </span>
                        )}
                      </span>
                    </div>
                  ))}
                  <div className="result-row">
                    <span className="result-label">Total change</span>
                    <span className="result-value">{optResult.total_change.toFixed(4)} m</span>
                  </div>
                  <div className="result-row">
                    <span className="result-label">Feasibility after</span>
                    <span className="result-value" style={{ color: "var(--success)" }}>
                      {(optResult.feasibility_ratio * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Right: viewer + chart ── */}
        <div>
          <RobotViewer trajectoryResult={trajResult} robotType={robotType}/>
          <p style={{ textAlign: "center", color: "var(--text2)", fontSize: "0.8rem", marginTop: "0.5rem" }}>
            🟢 Reachable &nbsp; 🔴 Unreachable
          </p>
          {trajResult && (
            <JointChart jointSequences={trajResult.joint_sequences} robotType={robotType}/>
          )}
        </div>
      </div>
    </div>
  );
}
