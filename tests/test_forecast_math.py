import math
import pytest

def test_return_to_price_identity():
    """Verify that return, target price, and price changes satisfy exact mathematical identities."""
    baseline = 10000.00
    pred_log_return = 0.000312

    # Mathematical formulas:
    pred_price = baseline * math.exp(pred_log_return)
    simple_return = math.exp(pred_log_return) - 1.0
    price_change = pred_price - baseline
    percent_change = 100.0 * simple_return

    # Check that price change is NOT zero
    assert price_change > 0, "Positive return must produce a positive price change"
    assert round(pred_price, 2) == 10003.12, f"Expected 10003.12, got {pred_price}"
    assert round(price_change, 2) == 3.12, f"Expected +3.12, got {price_change}"
    assert round(percent_change, 3) == 0.031, f"Expected +0.031%, got {percent_change}"

    # Verify identity: percent_change == 100 * (price_change / baseline)
    computed_pct = 100.0 * (price_change / baseline)
    assert abs(percent_change - computed_pct) < 1e-9

def test_negative_return_to_price():
    """Verify negative return conversion."""
    baseline = 45000.00
    pred_log_return = -0.005000

    pred_price = baseline * math.exp(pred_log_return)
    simple_return = math.exp(pred_log_return) - 1.0
    price_change = pred_price - baseline
    percent_change = 100.0 * simple_return

    assert price_change < 0
    assert pred_price < baseline
    assert abs(percent_change - (100.0 * price_change / baseline)) < 1e-9
