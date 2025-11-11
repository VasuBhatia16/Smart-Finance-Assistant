import { useMemo, useState } from "react";
import api from "./lib/api";
import "./App.css";

type Categories = Record<string, number>;

type MonthlyRecord = {
  month: string;          // YYYY-MM
  income: number;
  savings_goal: number;
  categories: Categories; // { rent: 12000, food: 7000, ... }
};

type ForecastPoint = {
  month: string;
  total_expense: number;
  projected_savings: number;
};

type PredictResponse = {
  forecast: ForecastPoint[];
  note?: string | null;
};

const DEFAULT_CATS: Categories = {
  rent: 12000,
  food: 7000,
  utilities: 3000,
  travel: 2000,
};

function sumCategories(cats: Categories): number {
  return Object.values(cats).reduce((a, b) => a + Number(b || 0), 0);
}

function isYYYYMM(v: string) {
  return /^\d{4}-\d{2}$/.test(v);
}

export default function App() {
  // ---- State: history list & editor form ----
  const [history, setHistory] = useState<MonthlyRecord[]>([
    { month: "2025-06", income: 60000, savings_goal: 15000, categories: { ...DEFAULT_CATS, shopping: 2500 } },
    { month: "2025-07", income: 60000, savings_goal: 15000, categories: { ...DEFAULT_CATS, food: 6500 } },
    { month: "2025-08", income: 60000, savings_goal: 15000, categories: { ...DEFAULT_CATS, shopping: 3500 } },
  ]);

  // editor fields for creating a new month row
  const [month, setMonth] = useState<string>("2025-09");
  const [income, setIncome] = useState<number>(60000);
  const [goal, setGoal] = useState<number>(15000);
  const [cats, setCats] = useState<Categories>({ ...DEFAULT_CATS });

  // UI control
  const [horizon, setHorizon] = useState<number>(3);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState<PredictResponse | null>(null);

  // ---- Derived values ----
  const totalOfEditor = useMemo(() => sumCategories(cats), [cats]);

  const canAdd =
    isYYYYMM(month) &&
    income >= 0 &&
    goal >= 0 &&
    totalOfEditor >= 0;

  const canPredict = history.length >= 3 && horizon >= 1 && horizon <= 12;

  // ---- Event handlers ----
  function handleAddCategoryRow() {
    // add a placeholder category name that doesn't clash
    let base = "other";
    let idx = 1;
    while (cats.hasOwnProperty(idx === 1 ? base : `${base}${idx}`)) idx++;
    const key = idx === 1 ? base : `${base}${idx}`;
    setCats({ ...cats, [key]: 0 });
  }

  function handleRemoveCategory(key: string) {
    const copy = { ...cats };
    delete copy[key];
    setCats(copy);
  }

  function handleChangeCategoryName(oldKey: string, newKey: string) {
    if (!newKey || newKey === oldKey) return;
    if (cats.hasOwnProperty(newKey)) {
      alert("Category name already exists.");
      return;
    }
    const copy: Categories = {};
    Object.entries(cats).forEach(([k, v]) => {
      copy[k === oldKey ? newKey : k] = v;
    });
    setCats(copy);
  }

  function handleChangeCategoryAmount(key: string, amt: number) {
    setCats(prev => ({ ...prev, [key]: isFinite(amt) ? amt : 0 }));
  }

  function handleAddMonth() {
    if (!canAdd) {
      alert("Please fix inputs (YYYY-MM month, non-negative numbers).");
      return;
    }
    setHistory(prev => [
      ...prev,
      {
        month,
        income,
        savings_goal: goal,
        categories: { ...cats },
      },
    ]);

    // next month helper: auto-advance the month field
    const [yStr, mStr] = month.split("-");
    const y = parseInt(yStr, 10);
    const m = parseInt(mStr, 10);
    const nextM = ((m % 12) + 1);
    const nextY = y + (m === 12 ? 1 : 0);
    setMonth(`${nextY.toString().padStart(4, "0")}-${nextM.toString().padStart(2, "0")}`);
  }

  function handleRemoveHistory(idx: number) {
    setHistory(prev => prev.filter((_, i) => i !== idx));
  }

  async function runPredict() {
    if (!canPredict) {
      alert("Need at least 3 months in history and 1–12 horizon.");
      return;
    }
    setLoading(true);
    setResp(null);
    try {
      const payload = { history, horizon };
      const r = await api.post<PredictResponse>("/predict", payload);
      setResp(r.data);
    } catch (e) {
      console.error(e);
      alert("Prediction failed. Check backend logs.");
    } finally {
      setLoading(false);
    }
  }

  function resetEditorToLastIncome() {
    const last = history[history.length - 1];
    if (last) {
      setIncome(last.income);
      setGoal(last.savings_goal);
    }
  }

  // ---- Render ----
  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Smart Personal Finance Assistant</h1>
        <span className="helper">Advanced input • backend-connected</span>
      </header>

      <main className="content">
        {/* Editor Card */}
        <div className="card" style={{ marginBottom: 16 }}>
          <h2 className="section-title">Add Monthly Record</h2>

          <div className="grid-3">
            <div>
              <label className="label">Month (YYYY-MM)</label>
              <input
                className="input"
                type="month"
                value={month}
                onChange={(e) => setMonth(e.target.value)}
              />
            </div>
            <div>
              <label className="label">Income (₹)</label>
              <input
                className="input"
                type="number"
                value={income}
                onChange={(e) => setIncome(Number(e.target.value))}
                min={0}
              />
            </div>
            <div>
              <label className="label">Savings Goal (₹)</label>
              <input
                className="input"
                type="number"
                value={goal}
                onChange={(e) => setGoal(Number(e.target.value))}
                min={0}
              />
            </div>
          </div>

          <div style={{ marginTop: 16 }}>
            <div className="row" style={{ alignItems: "center", justifyContent: "space-between" }}>
              <h3 className="section-title" style={{ margin: 0 }}>Categories</h3>
              <div className="helper">
                Total: <span className="badge">₹ {totalOfEditor.toLocaleString("en-IN")}</span>
              </div>
            </div>

            {/* Category rows */}
            <div className="row" style={{ flexDirection: "column" }}>
              {Object.entries(cats).map(([key, val]) => (
                <div key={key} className="cat-row">
                  <input
                    className="input cat-name"
                    value={key}
                    onChange={(e) => handleChangeCategoryName(key, e.target.value.trim())}
                    placeholder="Category name"
                  />
                  <input
                    className="input cat-amt"
                    type="number"
                    value={val}
                    min={0}
                    onChange={(e) => handleChangeCategoryAmount(key, Number(e.target.value))}
                    placeholder="Amount"
                  />
                  <button className="btn btn-ghost" onClick={() => handleRemoveCategory(key)}>
                    Remove
                  </button>
                </div>
              ))}
            </div>

            <div className="row" style={{ marginTop: 12 }}>
              <button className="btn btn-ghost" onClick={handleAddCategoryRow}>
                + Add Category
              </button>
              <span className="helper">
                Tip: Click on a category name to rename it. Use <span className="kbd">Tab</span> to move quickly.
              </span>
            </div>
          </div>

          <div className="row" style={{ marginTop: 16 }}>
            <button className="btn" onClick={handleAddMonth} disabled={!canAdd}>
              Add Month to History
            </button>
            <button className="btn btn-ghost" onClick={resetEditorToLastIncome}>
              Use Last Income/Goal
            </button>
          </div>
        </div>

        {/* History Table */}
        <div className="card" style={{ marginBottom: 16 }}>
          <h2 className="section-title">History</h2>
          {history.length === 0 ? (
            <p className="helper">No records yet. Add at least three months to enable forecasting.</p>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Month</th>
                    <th className="num">Income (₹)</th>
                    <th className="num">Savings Goal (₹)</th>
                    <th className="num">Total Expense (₹)</th>
                    <th>Top Categories</th>
                    <th className="num">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((h, idx) => {
                    const total = sumCategories(h.categories);
                    const topCats = Object.entries(h.categories)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 3)
                      .map(([k, v]) => `${k}: ₹${v.toLocaleString("en-IN")}`)
                      .join(", ");
                    return (
                      <tr key={idx}>
                        <td>{h.month}</td>
                        <td className="num">{h.income.toLocaleString("en-IN")}</td>
                        <td className="num">{h.savings_goal.toLocaleString("en-IN")}</td>
                        <td className="num">{total.toLocaleString("en-IN")}</td>
                        <td>{topCats || <span className="helper">—</span>}</td>
                        <td className="num">
                          <button className="btn btn-danger" onClick={() => handleRemoveHistory(idx)}>
                            Delete
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div className="row" style={{ marginTop: 16, alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <label className="label" style={{ margin: 0 }}>Horizon (months)</label>
              <input
                className="input"
                type="number"
                min={1}
                max={12}
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                style={{ width: 90 }}
              />
            </div>
            <button className="btn" onClick={runPredict} disabled={!canPredict || loading}>
              {loading ? "Predicting..." : "Run Forecast"}
            </button>
            <span className="helper">Need at least 3 months in history.</span>
          </div>
        </div>

        {/* Prediction Results */}
        <div className="card">
          <h2 className="section-title">Forecast</h2>
          {!resp ? (
            <p className="helper">No forecast yet. Add months and click “Run Forecast”.</p>
          ) : resp.forecast.length === 0 ? (
            <p className="helper">{resp.note || "No results."}</p>
          ) : (
            <>
              <div className="helper" style={{ marginBottom: 8 }}>{resp.note}</div>
              <div style={{ overflowX: "auto" }}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>Month</th>
                      <th className="num">Projected Expense (₹)</th>
                      <th className="num">Projected Savings (₹)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {resp.forecast.map((f) => (
                      <tr key={f.month}>
                        <td>{f.month}</td>
                        <td className="num">{f.total_expense.toLocaleString("en-IN")}</td>
                        <td className="num">{f.projected_savings.toLocaleString("en-IN")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
