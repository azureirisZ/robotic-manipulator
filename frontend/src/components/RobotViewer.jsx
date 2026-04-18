/**
 * RobotViewer — SVG-based 2D visualizer for planar RRR and PPP arms.
 * Draws the arm links, joints, end-effector, and optionally the trajectory path.
 */

const WIDTH  = 500;
const HEIGHT = 400;
const CX     = WIDTH  / 2;
const CY     = HEIGHT / 2;
const SCALE  = 90;   // pixels per meter

function worldToCanvas(x, y) {
  return [CX + x * SCALE, CY - y * SCALE];
}

export default function RobotViewer({ fkResult, trajectoryResult, robotType = "RRR" }) {
  const hasArm = fkResult && fkResult.joint_positions;

  // Build link segments from joint positions
  const joints = hasArm
    ? fkResult.joint_positions.map(([x, y]) => worldToCanvas(x, y))
    : null;

  // Trajectory path points
  const pathPoints = trajectoryResult?.path_points || [];
  const failedSet  = new Set(trajectoryResult?.failed_indices || []);

  return (
    <div className="viz-container">
      <svg width="100%" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} style={{ display: "block" }}>
        {/* Background grid */}
        <defs>
          <pattern id="grid" width={SCALE} height={SCALE} patternUnits="userSpaceOnUse"
            patternTransform={`translate(${CX % SCALE}, ${CY % SCALE})`}>
            <path d={`M ${SCALE} 0 L 0 0 0 ${SCALE}`} fill="none" stroke="#2e3250" strokeWidth="0.5"/>
          </pattern>
        </defs>
        <rect width={WIDTH} height={HEIGHT} fill="#1a1d27"/>
        <rect width={WIDTH} height={HEIGHT} fill="url(#grid)" opacity="0.6"/>

        {/* Axes */}
        <line x1={CX} y1={0} x2={CX} y2={HEIGHT} stroke="#2e3250" strokeWidth="1"/>
        <line x1={0} y1={CY} x2={WIDTH}  y2={CY} stroke="#2e3250" strokeWidth="1"/>
        <text x={CX + 6} y={14} fill="#4a5080" fontSize="11">Y</text>
        <text x={WIDTH - 16} y={CY - 6} fill="#4a5080" fontSize="11">X</text>

        {/* Workspace circle */}
        {fkResult?.link_lengths && (
          <circle
            cx={CX} cy={CY}
            r={(fkResult.link_lengths[0] + fkResult.link_lengths[1] + fkResult.link_lengths[2]) * SCALE}
            fill="none" stroke="#6c63ff" strokeWidth="1" strokeDasharray="6 4" opacity="0.25"
          />
        )}

        {/* Trajectory path */}
        {pathPoints.length > 0 && (
          <g>
            {pathPoints.map(([px, py], i) => {
              const [cx, cy] = worldToCanvas(px, py);
              const failed = failedSet.has(i);
              return (
                <circle key={i} cx={cx} cy={cy} r={3}
                  fill={failed ? "#ff5c5c" : "#00d4aa"} opacity={0.7}
                />
              );
            })}
          </g>
        )}

        {/* Robot arm */}
        {joints && (
          <g>
            {/* Links */}
            {joints.slice(0, -1).map(([x1, y1], i) => {
              const [x2, y2] = joints[i + 1];
              return (
                <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
                  stroke="#6c63ff" strokeWidth={i === 0 ? 6 : i === 1 ? 5 : 4}
                  strokeLinecap="round"
                />
              );
            })}

            {/* Link labels */}
            {joints.slice(0, -1).map(([x1, y1], i) => {
              const [x2, y2] = joints[i + 1];
              const mx = (x1 + x2) / 2;
              const my = (y1 + y2) / 2;
              return (
                <text key={i} x={mx + 6} y={my - 6} fill="#8b90b0" fontSize="11">
                  {robotType === "RRR" ? `a${i + 1}` : `d${i + 1}`}
                </text>
              );
            })}

            {/* Joints */}
            {joints.slice(0, -1).map(([cx, cy], i) => (
              <circle key={i} cx={cx} cy={cy} r={i === 0 ? 10 : 7}
                fill={i === 0 ? "#22263a" : "#1a1d27"}
                stroke={i === 0 ? "#6c63ff" : "#8b90b0"}
                strokeWidth={i === 0 ? 3 : 2}
              />
            ))}

            {/* Joint numbers */}
            {joints.slice(0, -1).map(([cx, cy], i) => (
              <text key={i} x={cx} y={cy + 4} textAnchor="middle"
                fill={i === 0 ? "#6c63ff" : "#8b90b0"} fontSize="9" fontWeight="700">
                J{i + 1}
              </text>
            ))}

            {/* End-effector */}
            {(() => {
              const [ex, ey] = joints[joints.length - 1];
              return (
                <g>
                  <circle cx={ex} cy={ey} r={9} fill="#00d4aa" opacity="0.2"/>
                  <circle cx={ex} cy={ey} r={5} fill="#00d4aa"/>
                  <text x={ex + 10} y={ey - 8} fill="#00d4aa" fontSize="11" fontWeight="600">EE</text>
                </g>
              );
            })()}
          </g>
        )}

        {/* Empty state */}
        {!hasArm && (
          <text x={CX} y={CY} textAnchor="middle" fill="#4a5080" fontSize="14">
            Configure robot and click Solve to visualize
          </text>
        )}

        {/* Legend */}
        <g transform={`translate(12, ${HEIGHT - 56})`}>
          <circle cx={6} cy={6}  r={4} fill="#6c63ff"/>
          <text x={14} y={10} fill="#8b90b0" fontSize="10">Joint</text>
          <circle cx={6} cy={22} r={4} fill="#00d4aa"/>
          <text x={14} y={26} fill="#8b90b0" fontSize="10">End-effector</text>
          <circle cx={6} cy={38} r={4} fill="#ff5c5c"/>
          <text x={14} y={42} fill="#8b90b0" fontSize="10">Unreachable point</text>
        </g>
      </svg>
    </div>
  );
}
