import { useState } from "react";
import axios from "axios";

interface Category {
  name: string;
  amount: number;
}

interface MonthRecord {
  month: string;
  income: number;
  savings_goal: number;
  categories: Category[];
}

interface ForecastItem {
  month: string;
  total_expense: number;
  projected_savings: number;
  category_breakdown: Record<string, number>;
}

export default function App() {
  const [history, setHistory] = useState<MonthRecord[]>([]);
  const [month, setMonth] = useState("");
  const [income, setIncome] = useState<number>(60000);
  const [savingsGoal, setSavingsGoal] = useState<number>(15000);
  const [categories, setCategories] = useState<Category[]>([]);
  const [modelType, setModelType] = useState("lstm");
  const [horizon, setHorizon] = useState(1);
  const [forecast, setForecast] = useState<ForecastItem[] | null>(null);

  const addCategory = () => {
    setCategories([...categories, { name: "", amount: 0 }]);
  };

  const updateCategory = (i: number, field: "name" | "amount", value: string) => {
    const updated = [...categories];
    if (field === "amount") {
      updated[i].amount = Number(value);
    } else {
      updated[i].name = value;
    }
    setCategories(updated);
  };

  const removeCategory = (i: number) => {
    setCategories(categories.filter((_, x) => x !== i));
  };

  const addMonth = () => {
    if (!month) return;

    setHistory([
      ...history,
      {
        month,
        income,
        savings_goal: savingsGoal,
        categories,
      },
    ]);

    setCategories([]);
    setMonth("");
  };

  const removeMonth = (index: number) => {
    setHistory(history.filter((_, i) => i !== index));
  };

  const runForecast = async () => {
    setForecast(null);

    try {
      const res = await axios.post("http://localhost:8000/api/predict", {
        history,
        horizon,
        model_type: modelType,
      });

      setForecast(res.data.forecast);
    } catch (e) {
      alert("Prediction failed");
      console.error(e);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 text-gray-900 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold tracking-tight">
            Smart Finance Assistant
          </h1>
        </div>

        {/* Control Panel */}
        <div className="bg-white shadow rounded-lg p-6 space-y-4">

          {/* Model + Horizon Panel */}
          <div className="flex gap-6 items-center">
            <div className="space-y-1">
              <label className="font-medium">Model</label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="border rounded px-3 py-2 w-48"
              >
                <option value="lstm">LSTM (Neural)</option>
                <option value="xgb">XGBoost</option>
              </select>
            </div>

            <div className="space-y-1">
              <label className="font-medium">Horizon</label>
              <input
                type="number"
                min={1}
                max={12}
                value={horizon}
                className="border px-3 py-2 rounded w-32"
                onChange={(e) => setHorizon(Number(e.target.value))}
              />
            </div>

            <button
              onClick={runForecast}
              className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-md font-medium"
            >
              Run Forecast
            </button>
          </div>
        </div>

        {/* Add Monthly Record */}
        <div className="bg-white shadow rounded-lg p-6 space-y-6">
          <h2 className="text-lg font-semibold">Add Monthly Record</h2>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="font-medium">Month (YYYY-MM)</label>
              <input
                value={month}
                onChange={(e) => setMonth(e.target.value)}
                placeholder="2025-09"
                className="border rounded px-3 py-2 w-full"
              />
            </div>

            <div>
              <label className="font-medium">Income</label>
              <input
                type="number"
                value={income}
                onChange={(e) => setIncome(Number(e.target.value))}
                className="border rounded px-3 py-2 w-full"
              />
            </div>

            <div>
              <label className="font-medium">Savings Goal</label>
              <input
                type="number"
                value={savingsGoal}
                onChange={(e) => setSavingsGoal(Number(e.target.value))}
                className="border rounded px-3 py-2 w-full"
              />
            </div>
          </div>

          {/* Categories */}
          <div className="space-y-3">
            <h3 className="font-medium">Categories</h3>

            {categories.map((c, i) => (
              <div className="flex items-center gap-4" key={i}>
                <input
                  value={c.name}
                  onChange={(e) => updateCategory(i, "name", e.target.value)}
                  placeholder="Category name"
                  className="border rounded px-3 py-2 w-64"
                />

                <input
                  type="number"
                  value={c.amount}
                  onChange={(e) => updateCategory(i, "amount", e.target.value)}
                  placeholder="Amount"
                  className="border rounded px-3 py-2 w-40"
                />

                <button
                  onClick={() => removeCategory(i)}
                  className="text-red-600 hover:text-red-700 text-sm font-medium"
                >
                  Remove
                </button>
              </div>
            ))}

            <button
              onClick={addCategory}
              className="bg-gray-200 hover:bg-gray-300 px-3 py-2 rounded-md"
            >
              Add Category
            </button>
          </div>

          <button
            onClick={addMonth}
            className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-md font-medium"
          >
            Add Month
          </button>
        </div>

        {/* History Table */}
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">History</h2>

          <table className="w-full border-collapse">
            <thead>
              <tr className="text-left bg-gray-50">
                <th className="p-3">Month</th>
                <th className="p-3">Income</th>
                <th className="p-3">Savings Goal</th>
                <th className="p-3">Total Expense</th>
                <th className="p-3">Top Categories</th>
                <th className="p-3">Actions</th>
              </tr>
            </thead>

            <tbody>
              {history.map((h, i) => {
                const total = h.categories.reduce((a, b) => a + b.amount, 0);
                const top = h.categories
                  .slice()
                  .sort((a, b) => b.amount - a.amount)
                  .slice(0, 3)
                  .map((c) => `${c.name}: ₹${c.amount}`)
                  .join(", ");

              return (
                <tr key={i} className="border-b">
                  <td className="p-3">{h.month}</td>
                  <td className="p-3">₹ {h.income.toLocaleString()}</td>
                  <td className="p-3">₹ {h.savings_goal.toLocaleString()}</td>
                  <td className="p-3">₹ {total.toLocaleString()}</td>
                  <td className="p-3">{top}</td>
                  <td className="p-3">
                    <button
                      onClick={() => removeMonth(i)}
                      className="text-red-600 hover:text-red-700 font-medium"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              );
            })}
            </tbody>
          </table>
        </div>

        {/* Forecast */}
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">Forecast</h2>

          {!forecast && <p>No forecast yet.</p>}

          {forecast && (
            <div className="space-y-4">
              {forecast.map((f, i) => (
                <div
                  key={i}
                  className="border rounded-lg p-4 bg-gray-50"
                >
                  <h3 className="font-medium">{f.month}</h3>

                  <p className="text-gray-700">
                    <strong>Total Expense:</strong> ₹{f.total_expense.toLocaleString()}
                  </p>
                  <p className="text-gray-700">
                    <strong>Projected Savings:</strong> ₹{f.projected_savings.toLocaleString()}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
