import { useState } from "react";
import "./App.css";

export default function App() {
  const [history, setHistory] = useState([
    {
      month: "",
      income: "",
      savings_goal: "",
      categories: [{ name: "", amount: "" }],
    },
  ]);

  const [horizon, setHorizon] = useState(1);
  const [modelType, setModelType] = useState("xgb");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const addCategory = (idx: number) => {
    const copy = [...history];
    copy[idx].categories.push({ name: "", amount: "" });
    setHistory(copy);
  };

  const removeCategory = (hidx: number, cidx: number) => {
    const copy = [...history];
    copy[hidx].categories.splice(cidx, 1);
    setHistory(copy);
  };

  const updateHistory = (idx: number, field: string, val: any) => {
    const copy = [...history];
    (copy[idx] as any)[field] = val;
    setHistory(copy);
  };

  const updateCategory = (hidx: number, cidx: number, field: string, val: any) => {
    const copy = [...history];
    (copy[hidx].categories[cidx] as any)[field] = val;
    setHistory(copy);
  };

  const addHistoryMonth = () => {
    setHistory([
      ...history,
      {
        month: "",
        income: "",
        savings_goal: "",
        categories: [{ name: "", amount: "" }],
      },
    ]);
  };

  const runForecast = async () => {
  setLoading(true);
  setResult(null);

  const cleanedHistory = history.map((h) => {
    const catDict: any = {};
    h.categories.forEach((c) => {
      if (c.name && c.amount) {
        catDict[c.name] = Number(c.amount);
      }
    });

    return {
      month: h.month,
      income: Number(h.income),
      savings_goal: Number(h.savings_goal),
      categories: catDict,
    };
  });

  try {
    const res = await fetch("http://localhost:8000/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        history: cleanedHistory,
        horizon,
        model_type: modelType,
      }),
    });

    const data = await res.json();

    // ❌ Do NOT set forecast before checking success
    if (!res.ok) {
      const msg = data?.detail?.[0]?.msg || "An error occurred";
      setResult({ error: msg });
      setLoading(false);
      return;
    }

    // ✔ SUCCESS
    setResult({
      forecast: data.forecast,
      note: data.note,
    });

  } catch (err) {
    setResult({
      error: "Prediction failed. Check your input or add at least 3 months.",
    });
  }

  setLoading(false);
};




  return (
    <div className="dashboard-root">
      {/* ---------------- SIDEBAR ---------------- */}
      <aside className="sidebar">
        <h3 className="sidebar-title">Smart Finance</h3>

        <div className="side-block">
          <label>Model Type</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
          >
            <option value="xgb">XGBoost</option>
            <option value="lstm">LSTM</option>
          </select>
        </div>

        <div className="side-block">
          <label>Forecast Horizon (months)</label>
          <input
            type="number"
            min="1"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
          />
        </div>

        <button className="run-btn" onClick={runForecast}>
          {loading ? "Running..." : "Generate Forecast"}
        </button>
      </aside>

      {/* ---------------- MAIN CONTENT ---------------- */}
      <main className="main-content">
        <h2 className="section-header">Historical Records</h2>

        {history.map((h, idx) => (
          <div className="card" key={idx}>
            <div className="card-header">Month {idx + 1}</div>

            <div className="input-grid">
              <div className="input-box">
                <label>Month (YYYY-MM)</label>
                <input
                  type="month"
                  value={h.month}
                  onChange={(e) => updateHistory(idx, "month", e.target.value)}
                />
              </div>

              <div className="input-box">
                <label>Income</label>
                <input
                  type="number"
                  value={h.income}
                  onChange={(e) => updateHistory(idx, "income", e.target.value)}
                />
              </div>

              <div className="input-box">
                <label>Savings Goal</label>
                <input
                  type="number"
                  value={h.savings_goal}
                  onChange={(e) =>
                    updateHistory(idx, "savings_goal", e.target.value)
                  }
                />
              </div>
            </div>

            <div className="category-title">Categories</div>

            {h.categories.map((c, cidx) => (
              <div className="category-row" key={cidx}>
                <input
                  placeholder="Category"
                  value={c.name}
                  onChange={(e) =>
                    updateCategory(idx, cidx, "name", e.target.value)
                  }
                />
                <input
                  type="number"
                  placeholder="Amount"
                  value={c.amount}
                  onChange={(e) =>
                    updateCategory(idx, cidx, "amount", e.target.value)
                  }
                />

                {h.categories.length > 1 && (
                  <button
                    className="remove-btn"
                    onClick={() => removeCategory(idx, cidx)}
                  >
                    ×
                  </button>
                )}
              </div>
            ))}

            <button className="add-btn" onClick={() => addCategory(idx)}>
              + Add Category
            </button>
          </div>
        ))}

        <button className="add-month-btn" onClick={addHistoryMonth}>
          + Add Another Month
        </button>

        {/* ---------------- RESULT SECTION ---------------- */}
        {result && (
  <div className="forecast-box">
    {result.error ? (
      <div className="error-box">
        <strong>Error:</strong> {result.error}
      </div>
    ) : (
      <div className="forecast-success">
        <h3>Forecast Result</h3>

        {Array.isArray(result.forecast) &&
          result.forecast.map((f: any, idx: number) => (
            <div className="forecast-item" key={idx}>
              <div><strong>Month:</strong> {f.month}</div>
              <div><strong>Total Expense:</strong> {f.total_expense}</div>
              <div><strong>Projected Savings:</strong> {f.projected_savings}</div>

              <div className="cat-break">
                <strong>Category Breakdown:</strong>
                <ul>
                  {Object.entries(f.category_breakdown as Record<string, number>).map(([cat, val]: [string, number]) => (
                    <li key={cat}>
                      {cat}: {val}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}

        {result.note && <p className="note">{result.note}</p>}
      </div>
    )}
  </div>
)}

      </main>
    </div>
  );
}
