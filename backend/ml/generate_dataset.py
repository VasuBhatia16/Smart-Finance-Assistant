#!/usr/bin/env python3
"""
realistic_smart_finance_data_generator.py

- Fixes CSV column ordering so Subcategory gets the correct value.
- Adds option to write Income amounts as negative (if downstream expects that).
- Optionally injects a monthly salary row for realistic income.
- Performs no external I/O other than writing the CSV file.
"""

import random
import csv
from datetime import datetime
import calendar

# ----------------------------
# CONFIGURABLE OPTIONS
# ----------------------------
INCOME_SIGN_AS_NEGATIVE = False    # If True, Income amounts will be written as negative numbers.
INJECT_MONTHLY_SALARY = True      # If True, append a guaranteed salary row each month.
SALARY_AMOUNT = 30000.00          # Amount for the injected monthly salary
ENTRIES_PER_MONTH = 120           # Random transactions per month (excluding injected salary)
OUTPUT_CSV = "realistic_smart_finance_data.csv"
# ----------------------------

CATEGORY_WEIGHTS = {
    "Food": 0.58,
    "Transportation": 0.15,
    "Other": 0.15,
    "Social Life": 0.05,
    "Household": 0.04,
    "Apparel": 0.02,
    "Beauty": 0.005,
    "Gift": 0.002,
    "Education": 0.003
}
# Note: Keep these weights as-is (they sum ~1). We do not add 'Income' here; salary is injected explicitly.

SUBCATEGORIES = {
    "Food": ["Lunch", "Dinner", "Snacks", "Breakfast", "Milk with bharath", "Kfc dinner", "Pizza", "Parotta", "Water", "Corn"],
    "Transportation": ["Rapido", "Auto", "Metro", "Bus ticket", "Taxi", "Train"],
    "Other": ["To vicky", "To gobi", "From dad", "From friend", "To kumara", "To barath", "To siva", "From stu"],
    "Social Life": ["Games", "Beer", "Movie", "Badminton"],
    "Household": ["Rent", "Stuffs", "Mirror"],
    "Apparel": ["Hoodie", "Full hand and hoodie", "Earphone", "Shoes", "Cap"],
    "Beauty": ["Showergel"],
    "Gift": ["Gift"],
    "Education": ["Arrear fee", "Books"]
}

NOTES = ["", "", "", "", "with company", "with bharath", "with gobi", "with stu"]

# A small set of explicit income-like subcategories for automatic detection (not part of category weights)
INCOME_LIKE_SUBS = {"Salary", "Stipend", "Refund", "Cashback", "From dad", "From friend", "From stu"}

def choose_category():
    """Choose a category using the CATEGORY_WEIGHTS dict (handles any sum by normalizing)."""
    total = sum(CATEGORY_WEIGHTS.values())
    r = random.random() * total
    cumulative = 0.0
    for cat, weight in CATEGORY_WEIGHTS.items():
        cumulative += weight
        if r <= cumulative:
            return cat
    # fallback
    return list(CATEGORY_WEIGHTS.keys())[0]

def random_amount(category, sub):
    """Return a realistic amount depending on category and subcategory."""
    if category == "Household" and sub == "Rent":
        return random.uniform(4500, 5000)

    if "From dad" in sub:
        return random.choice([500, 1000, 1500, 2000, 8000, 10000])

    if "From" in sub and sub not in INCOME_LIKE_SUBS:
        return random.randint(20, 400)

    if category == "Apparel":
        return random.uniform(300, 1800)

    if category == "Social Life":
        return random.uniform(100, 1800)

    if category == "Transportation":
        return random.uniform(40, 1600)

    if category == "Education":
        return random.uniform(300, 1500)

    # a mix of small, medium, and larger random values for food/other
    return random.choice([
        random.uniform(10, 100),
        random.uniform(80, 300),
        random.uniform(250, 700)
    ])

def random_datetime(year, month):
    hour_pool = [8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22]
    day = random.randint(1, calendar.monthrange(year, month)[1])
    hour = random.choice(hour_pool)
    minute = random.randint(0, 59)
    return datetime(year, month, day, hour, minute)

def generate_month(year, month, entries_per_month=ENTRIES_PER_MONTH):
    """Generate rows for a single month. Returns a list of rows (each row is a list matching header)."""
    rows = []
    for _ in range(entries_per_month):
        dt = random_datetime(year, month)

        # Weekend behaviour: more Food & Social Life
        if dt.weekday() >= 5:
            category = random.choices(
                population=["Food", "Social Life", "Transportation", "Other"],
                weights=[0.7, 0.15, 0.10, 0.05]
            )[0]
        else:
            category = choose_category()

        # choose subcategory from the proper category (guaranteed non-empty)
        sub = random.choice(SUBCATEGORIES[category])

        amt = random_amount(category, sub)
        amount_val = round(amt, 2)

        # Decide Income vs Expense:
        if any(marker in sub for marker in ("From",)) or sub in INCOME_LIKE_SUBS:
            inc_exp = "Income"
        else:
            inc_exp = "Expense"

        # Apply sign convention if requested
        if inc_exp == "Income" and INCOME_SIGN_AS_NEGATIVE:
            amount_for_csv = -amount_val
        else:
            amount_for_csv = amount_val

        # If rent, enforce a stable day range early in month
        if category == "Household" and "Rent" in sub:
            dt = dt.replace(day=random.randint(3, 10))

        # NOTE: Correct column ordering: Subcategory receives `sub`. Note placed in Note column.
        row = [
            dt.strftime("%m/%d/%Y %H:%M"),  # Date
            "CUB - online payment",        # Account
            category,                      # Category
            sub,                           # Subcategory  <- CORRECTED
            random.choice(NOTES),          # Note
            amount_for_csv,                # INR (numeric)
            inc_exp,                       # Income/Expense
            "",                            # Note2 (left empty)
            amount_for_csv,                # Amount (canonical numeric)
            "INR",                         # Currency
            ""                             # Account2 (left empty)
        ]
        rows.append(row)

    # Optionally inject one monthly salary row to ensure regular income
    if INJECT_MONTHLY_SALARY:
        salary_day = random.randint(1, 5)
        salary_dt = datetime(year, month, salary_day, 10, 0)
        salary_amount = SALARY_AMOUNT
        salary_signed = -salary_amount if INCOME_SIGN_AS_NEGATIVE else salary_amount
        salary_row = [
            salary_dt.strftime("%m/%d/%Y %H:%M"),
            "CUB - salary",
            "Other",                # or "Income" depending on your downstream; keep "Other" to fit existing weights
            "Salary",
            "Monthly salary",
            salary_signed,
            "Income",
            "",
            salary_signed,
            "INR",
            ""
        ]
        rows.append(salary_row)

    return rows

def generate_dataset(start_year=2022, start_month=3, end_year=2025, end_month=10, entries_per_month=ENTRIES_PER_MONTH):
    all_rows = []
    year, month = start_year, start_month
    while True:
        print(f"Generating: {month}/{year}")
        all_rows.extend(generate_month(year, month, entries_per_month))
        # advance month
        month += 1
        if month > 12:
            month = 1
            year += 1
        if year > end_year or (year == end_year and month > end_month):
            break
    return all_rows

def write_csv(rows, output_file=OUTPUT_CSV):
    header = ["Date", "Account", "Category", "Subcategory", "Note", "INR",
              "Income/Expense", "Note2", "Amount", "Currency", "Account2"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"✅ {output_file} generated")
    print("✅ Total rows:", len(rows))

def quick_sanity_check(output_file=OUTPUT_CSV, look_at_month=None):
    """
    Print simple sanity stats:
    - counts by Category (top 10)
    - count of Income vs Expense
    - monthly sum (basic) for a sample month if requested (format 'YYYY-MM' or None)
    Uses only csv + built-ins so no external dependencies are required.
    """
    import csv
    from collections import Counter, defaultdict
    cat_counter = Counter()
    ie_counter = Counter()
    monthly_sums = defaultdict(float)

    with open(output_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cat = r["Category"]
            sub = r["Subcategory"]
            ie = r["Income/Expense"]
            try:
                amt = float(r["Amount"])
            except Exception:
                amt = 0.0
            cat_counter[cat] += 1
            ie_counter[ie] += 1
            # extract YYYY-MM from Date (source format 'MM/DD/YYYY HH:MM')
            try:
                date_part = r["Date"].split()[0]
                mm, dd, yyyy = date_part.split("/")
                ymd = f"{yyyy}-{mm.zfill(2)}"
                monthly_sums[ymd] += amt
            except Exception:
                pass

    print("\nCategory counts (top):")
    for k, v in cat_counter.most_common(10):
        print(f"  {k}: {v}")
    print("\nIncome/Expense counts:")
    for k, v in ie_counter.items():
        print(f"  {k}: {v}")

    if look_at_month:
        print(f"\nMonthly sum for {look_at_month}: {monthly_sums.get(look_at_month, 0.0)}")
    else:
        # print last 6 months sums (sorted)
        keys = sorted(monthly_sums.keys())[-6:]
        print("\nLast up to 6 monthly totals:")
        for k in keys:
            print(f"  {k}: {monthly_sums[k]}")

if __name__ == "__main__":
    rows = generate_dataset(entries_per_month=ENTRIES_PER_MONTH)
    write_csv(rows, OUTPUT_CSV)
    # quick sanity check (prints to console)
    quick_sanity_check(OUTPUT_CSV)
