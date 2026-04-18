import { useState } from "react";
import { solveFK, solveIK } from "../api/client";
import RobotViewer from "../components/RobotViewer";

const DEFAULT_LINKS = [1.0, 0.8, 0.5];

export default function Simulator() {
  const [robotType, setRobotType]     = useState("RRR");
  const [mode, setMode]               = useState("fk");   // "fk" or "ik"
  const [links, setLinks]             = useState(DEFAULT_LINKS);
  const [jointVals, setJointVals]     = useState([0, 0, 0]);
  const [target, setTarget]           = useState({ x: 1.2, y: 0.5, z: 0 });
  const [elbowUp, setElbowUp]         = useState(true);
  const [result, setResult]           = useState(null);
  const [fkResult, setFkResult]       = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState("");

  const handleSolve = async () => {
    setLoading(true);
    setError("");
    try {
      if (mode === "fk") {
        const res = await solveFK(robotType, jointVals, robotType === "RRR" ? links : null);
        setResult({ mode: "fk", data: res.data });
        setFkResult({ ...res.data, link_lengths: links });
      } else {
        const res = await solveIK(robotType, target.x, target.y, target.z, robotType === "RRR" ? links : null, elbowUp);
        setResult({ mode: "ik", data: res.data });
        if (res.data.success) {
          // Run FK with the IK result to get joint positions for visualization
          const fk = await solveFK(robotType, res.data.joint_values, robotType === "RRR" ? links : null);
          setFkResult({ ...fk.data, link_lengths: links });
        }
      }
    } catch (e) {
      setError(e.response?.data?.detail || e.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const updateLink = (i, val) => {
    const updated = [...links];
    updated[i] = parseFloat(val) || 0;
    setLinks(updated);
  };

  const updateJoint = (i, val) => {
    const updated = [...jointVals];
    updated[i] = parseFloat(val) || 0;
    setJointVals(updated);
  };

  return (
    <div>
      <div className="page-header">
        <h1>Robot Simulator</h1>
        <p>Solve forward and inverse kinematics for a single end-effector position</p>
      </div>

      <div className="grid-2">
        {/* ── Left panel ── */}
        <div>
          {/* Robot type */}
          <div className="card" style={{ marginBottom: "1rem" }}>
            <div className="card-title">Robot Configuration</div>

            <div className="toggle-group">
              {["RRR", "PPP"].map((t) => (
                <button key={t} className={`toggle-btn ${robotType === t ? "active" : ""}`}
                  onClick={() => { setRobotType(t); setResult(null); setFkResult(null); }}>
                  {t}
                </button>
              ))}
            </div>

            {robotType === "RRR" && (
              <div>
                <div className="form-group">
                  <label>Link Lengths (m)</label>
                  <div className="form-row">
                    {["a1", "a2", "a3"].map((label, i) => (
                      <div key={i}>
                        <label style={{ fontSize: "0.75rem", color: "var(--accent)" }}>{label}</label>
                        <input type="number" step="0.1" min="0.1"
                          value={links[i]}
                          onChange={(e) => updateLink(i, e.target.value)}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Mode */}
          <div className="card" style={{ marginBottom: "1rem" }}>
            <div className="card-title">Solver Mode</div>
            <div className="toggle-group">
              <button className={`toggle-btn ${mode === "fk" ? "active" : ""}`} onClick={() => setMode("fk")}>
                Forward Kinematics
              </button>
              <button className={`toggle-btn ${mode === "ik" ? "active" : ""}`} onClick={() => setMode("ik")}>
                Inverse Kinematics
              </button>
            </div>

            {mode === "fk" ? (
              <div>
                <div className="form-group">
                  <label>{robotType === "RRR" ? "Joint Angles (rad)" : "Prismatic Displacements (m)"}</label>
                  <div className="form-row">
                    {(robotType === "RRR" ? ["θ1", "θ2", "θ3"] : ["d1", "d2", "d3"]).map((label, i) => (
                      <div key={i}>
                        <label style={{ fontSize: "0.75rem", color: "var(--accent2)" }}>{label}</label>
                        <input type="number" step="0.1"
                          value={jointVals[i]}
                          onChange={(e) => updateJoint(i, e.target.value)}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div>
                <div className="form-group">
                  <label>Target Position (m)</label>
                  <div className="form-row">
                    {["x", "y", "z"].map((axis) => (
                      <div key={axis}>
                        <label style={{ fontSize: "0.75rem", color: "var(--accent2)" }}>{axis.toUpperCase()}</label>
                        <input type="number" step="0.1"
                          value={target[axis]}
                          onChange={(e) => setTarget({ ...target, [axis]: parseFloat(e.target.value) || 0 })}
                        />
                      </div>
                    ))}
                  </div>
                </div>
                {robotType === "RRR" && (
                  <div className="toggle-group">
                    <button className={`toggle-btn ${elbowUp ? "active" : ""}`} onClick={() => setElbowUp(true)}>
                      Elbow Up
                    </button>
                    <button className={`toggle-btn ${!elbowUp ? "active" : ""}`} onClick={() => setElbowUp(false)}>
                      Elbow Down
                    </button>
                  </div>
                )}
              </div>
            )}

            <button className="btn btn-primary" onClick={handleSolve} disabled={loading}>
              {loading ? "Solving..." : "Solve"}
            </button>

            {error && <div className="error-msg">{error}</div>}
          </div>

          {/* Results */}
          {result && (
            <div className="card">
              <div className="card-title">Results</div>

              {result.mode === "fk" && (
                <>
                  <div className="result-row">
                    <span className="result-label">Position X</span>
                    <span className="result-value">{result.data.end_effector.x.toFixed(4)} m</span>
                  </div>
                  <div className="result-row">
                    <span className="result-label">Position Y</span>
                    <span className="result-value">{result.data.end_effector.y.toFixed(4)} m</span>
                  </div>
                  <div className="result-row">
                    <span className="result-label">Position Z</span>
                    <span className="result-value">{result.data.end_effector.z.toFixed(4)} m</span>
                  </div>
                  {robotType === "RRR" && (
                    <div className="result-row">
                      <span className="result-label">Yaw</span>
                      <span className="result-value">{(result.data.end_effector.yaw * 180 / Math.PI).toFixed(2)}°</span>
                    </div>
                  )}
                </>
              )}

              {result.mode === "ik" && (
                <>
                  <div className="result-row">
                    <span className="result-label">Status</span>
                    <span className={`badge ${result.data.success ? "badge-success" : "badge-danger"}`}>
                      {result.data.success ? "✓ Reachable" : "✗ Unreachable"}
                    </span>
                  </div>
                  {result.data.success && result.data.joint_values_deg.map((v, i) => (
                    <div className="result-row" key={i}>
                      <span className="result-label">
                        {robotType === "RRR" ? `θ${i + 1}` : `d${i + 1}`}
                      </span>
                      <span className="result-value">
                        {robotType === "RRR" ? `${v.toFixed(2)}°` : `${v.toFixed(4)} m`}
                      </span>
                    </div>
                  ))}
                  <div className="result-row">
                    <span className="result-label">Config</span>
                    <span className="result-value" style={{ color: "var(--accent2)", fontSize: "0.82rem" }}>
                      {result.data.configuration}
                    </span>
                  </div>
                  {result.data.error !== undefined && (
                    <div className="result-row">
                      <span className="result-label">Position error</span>
                      <span className="result-value">{result.data.error.toExponential(2)} m</span>
                    </div>
                  )}
                  {result.data.alternate_solution?.success && (
                    <div style={{ marginTop: "1rem" }}>
                      <div className="card-title">Alternate Solution ({result.data.alternate_solution.configuration})</div>
                      {result.data.alternate_solution.joint_values_deg.map((v, i) => (
                        <div className="result-row" key={i}>
                          <span className="result-label">θ{i + 1}</span>
                          <span className="result-value" style={{ color: "var(--text2)" }}>{v.toFixed(2)}°</span>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        {/* ── Right panel: visualizer ── */}
        <div>
          <RobotViewer fkResult={fkResult} robotType={robotType}/>
          <p style={{ textAlign: "center", color: "var(--text2)", fontSize: "0.8rem", marginTop: "0.5rem" }}>
            2D workspace view — dashed circle = max reach
          </p>
        </div>
      </div>
    </div>
  );
}
