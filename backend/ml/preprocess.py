import numpy as np

ESSENTIALS = {"rent", "utilities", "bills", "emi"}

def extract_features(record):
    """
    record: MonthlyRecord-like dict
    returns: fixed-size numpy array of features
    """
    income = float(record["income"])
    goal = float(record["savings_goal"])
    cats = record["categories"]

    total_exp = sum(float(v) for v in cats.values())

    essential_exp = sum(float(v) for k, v in cats.items() if k.lower() in ESSENTIALS)
    discretionary_exp = max(total_exp - essential_exp, 0.0)

    ratio = discretionary_exp / total_exp if total_exp > 0 else 0.0

    return np.array([
        income,
        goal,
        total_exp,
        essential_exp,
        discretionary_exp,
        ratio,
    ], dtype=np.float32)


def build_feature_matrix(history):
    """
    history: list of dicts representing monthly records
    returns shape (T, F)
    """
    return np.stack([extract_features(r) for r in history], axis=0)
