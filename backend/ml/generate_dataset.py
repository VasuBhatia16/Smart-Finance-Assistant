import random
import csv
from datetime import datetime, timedelta
import calendar


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


def choose_category():
    r = random.random()
    cumulative = 0
    for cat, weight in CATEGORY_WEIGHTS.items():
        cumulative += weight
        if r <= cumulative:
            return cat
    return "Food"

def random_amount(category, sub):
    if category == "Household" and sub == "Rent":
        return random.uniform(4500, 5000)

    if "From dad" in sub:
        return random.choice([500, 1000, 1500, 2000, 8000, 10000])

    if "From" in sub:
        return random.randint(20, 400)

    if category == "Apparel":
        return random.uniform(300, 1800)

    if category == "Social Life":
        return random.uniform(100, 1800)

    if category == "Transportation":
        return random.uniform(40, 1600)

    if category == "Education":
        return random.uniform(300, 1500)

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

def generate_month(year, month, entries_per_month=120):
    rows = []
    for _ in range(entries_per_month):
        dt = random_datetime(year, month)

        if dt.weekday() >= 5: 
            category = random.choices(
                population=["Food", "Social Life", "Transportation", "Other"],
                weights=[0.7, 0.15, 0.10, 0.05]
            )[0]
        else:
            category = choose_category()

        sub = random.choice(SUBCATEGORIES[category])

        amt = random_amount(category, sub)

        inc_exp = "Income" if "From" in sub else "Expense"

        if category == "Household" and "Rent" in sub:
            dt = dt.replace(day=random.randint(3, 10))

        row = [
            dt.strftime("%m/%d/%Y %H:%M"),
            "CUB - online payment",
            category,
            "",
            sub,
            round(amt, 2),
            inc_exp,
            random.choice(NOTES),
            round(amt, 2),
            "INR",
            round(amt, 2)
        ]
        rows.append(row)

    return rows


def generate_dataset(start_year=2022, start_month=3, end_year=2025, end_month=10, entries_per_month=120):
    all_rows = []
    year, month = start_year, start_month

    while True:
        print(f"Generating: {month}/{year}")
        all_rows.extend(generate_month(year, month, entries_per_month))

        month += 1
        if month > 12:
            month = 1
            year += 1

        if year > end_year or (year == end_year and month > end_month):
            break

    return all_rows


if __name__ == "__main__":
    rows = generate_dataset(entries_per_month=120) 

    header = ["Date", "Account", "Category", "Subcategory", "Note", "INR",
              "Income/Expense", "Note2", "Amount", "Currency", "Account2"]

    with open("realistic_smart_finance_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("✅ realistic_smart_finance_data.csv generated")
    print("✅ Total rows:", len(rows))
