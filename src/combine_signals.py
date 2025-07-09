def final_decision(stock_pred: str, etf_pred: str, stock_conf: float, etf_conf: float) -> str:
    """
    Combine signals from stock and ETF models to make a final decision.

    - Prioritize high-confidence short-term trades
    - Use long-term ETF trend to adjust risk or filter noise

    Returns:
        str: Decision ("ENTER_TRADE", "RISKY_TRADE", "EXIT_OR_HEDGE", "WAIT")
    """
    if stock_pred == "UP" and stock_conf >= 0.80:
        return "ENTER_TRADE"  # 🔥 strong confidence, override filter
    elif stock_pred == "UP" and etf_pred == "UP":
        return "ENTER_TRADE"
    elif stock_pred == "UP":
        return "RISKY_TRADE"
    elif stock_pred == "DOWN":
        return "EXIT_OR_HEDGE"
    else:
        return "WAIT"
