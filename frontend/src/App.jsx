import { useState } from "react";
import { getIK } from "./api/client";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleIK = async () => {
    setLoading(true);
    try {
      const res = await getIK({
        x: 1.2,
        y: 0.5,
        link_lengths: [1.0, 0.8, 0.5],
        elbow_up: true
      });
      setResult(res.data);
    } catch (err) {
      setResult({ error: "API error" });
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 text-gray-800">

      {/* Header */}
      <div className="bg-white shadow p-4 text-xl font-semibold">
        Robotic Manipulator Simulator
      </div>

      {/* Main */}
      <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">

        {/* Control Panel */}
        <div className="bg-white p-6 rounded-2xl shadow">
          <h2 className="text-lg font-semibold mb-4">Controls</h2>

          <button
            onClick={handleIK}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition"
          >
            {loading ? "Computing..." : "Run Inverse Kinematics"}
          </button>
        </div>

        {/* Results */}
        <div className="bg-white p-6 rounded-2xl shadow">
          <h2 className="text-lg font-semibold mb-4">Results</h2>

          {result ? (
            <pre className="text-sm bg-gray-100 p-4 rounded overflow-auto">
              {JSON.stringify(result, null, 2)}
            </pre>
          ) : (
            <p className="text-gray-500">No data yet</p>
          )}
        </div>

      </div>
    </div>
  );
}
