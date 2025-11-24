# utils/formatter.py

def format_mse(mse: float) -> str:
    """Format MSE as scientific notation with two decimal places"""
    return f"{mse:.2e}"

def format_time(time_val: float) -> str:
    """Format time with two decimal places"""
    return round(time_val, 2)