import pandas as pd
import json

df = pd.read_csv("data-synthetic.csv")

df["Date"] = pd.to_datetime(df["Date"])

df["month"] = df["Date"].dt.to_period("M").astype(str)

map_to_global = {
    "Rent": "rent",
    "House Rent": "rent",
    "Food": "food",
    "Groceries": "food",
    "Bills": "utilities",
    "Electricity": "utilities",
    "Travel": "travel",
    "Shopping": "shopping",
    "Education": "education",
    "Health": "health",
}

def normalize_category(cat):
    for k, v in map_to_global.items():
        if str(cat).lower().startswith(k.lower()):
            return v
    return "misc"


df["norm_cat"] = df["Category"].apply(normalize_category)

monthly_groups = df.groupby(["month", "norm_cat"])["Amount"].sum().reset_index()

pivot = monthly_groups.pivot(index="month", columns="norm_cat", values="Amount").fillna(0)

history = []

for month, row in pivot.iterrows():
    record = {
        "month": month,
        "income": 60000,
        "savings_goal": 15000,
        "categories": row.to_dict()
    }
    history.append(record)

with open("training_data.json", "w") as f:
    json.dump(history, f, indent=2)

print("training_data.json generated!")
