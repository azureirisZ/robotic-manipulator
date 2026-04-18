import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from "recharts";

export default function JointChart({ jointSequences, robotType = "RRR" }) {
  if (!jointSequences) return null;

  const keys   = Object.keys(jointSequences);
  const colors = ["#6c63ff", "#00d4aa", "#f5a623"];
  const n      = jointSequences[keys[0]]?.length || 0;

  if (n === 0) return null;

  const data = Array.from({ length: n }, (_, i) => {
    const point = { index: i };
    keys.forEach((k) => {
      const v = jointSequences[k]?.[i];
      point[k] = v !== null && v !== undefined
        ? (robotType === "RRR" ? parseFloat((v * 180 / Math.PI).toFixed(2)) : parseFloat(v.toFixed(4)))
        : null;
    });
    return point;
  });

  const unit = robotType === "RRR" ? "°" : "m";

  return (
    <div className="card" style={{ marginTop: "1.5rem" }}>
      <div className="card-title">Joint Variables Along Trajectory</div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2e3250"/>
          <XAxis dataKey="index" stroke="#8b90b0" tick={{ fontSize: 11 }} label={{ value: "Point index", position: "insideBottom", offset: -2, fill: "#8b90b0", fontSize: 11 }}/>
          <YAxis stroke="#8b90b0" tick={{ fontSize: 11 }} unit={unit}/>
          <Tooltip
            contentStyle={{ background: "#22263a", border: "1px solid #2e3250", borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: "#8b90b0" }}
            formatter={(v, name) => [`${v}${unit}`, name]}
          />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }}/>
          {keys.map((k, i) => (
            <Line
              key={k}
              type="monotone"
              dataKey={k}
              stroke={colors[i]}
              strokeWidth={2}
              dot={false}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
