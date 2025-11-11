def add_months(year: int, month: int, n: int):
    """
    Add n months to given (year, month)
    Returns new (year, month)
    """
    new_month = month + n
    new_year = year + (new_month - 1) // 12
    new_month = ((new_month - 1) % 12) + 1
    return new_year, new_month
