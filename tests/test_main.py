"""Testes de main.py: unitario (_add_expected_value) e de integracao (simulate_roi)."""

import pandas as pd

from main import _add_expected_value, simulate_roi


def test_add_expected_value_flags_profitable_offers():
    scored_offers = pd.DataFrame(
        {
            "offer_type": ["bogo", "discount", "informational"],
            "min_value": [10, 10, 0],
            "discount_value": [10, 2, 0],
            "y_pred_proba": [0.9, 0.9, 0.9],
            "success": [1, 1, 1],
        }
    )

    result = _add_expected_value(scored_offers, margin_rate=0.3, send_cost=0.05)

    assert "informational" not in result["offer_type"].values
    bogo_row = result[result["offer_type"] == "bogo"].iloc[0]
    discount_row = result[result["offer_type"] == "discount"].iloc[0]
    # bogo devolve 100% do min_value -> so lucrativo com margem >= 100%, nao 30%
    assert bool(bogo_row["send_recommended"]) is False
    assert bool(discount_row["send_recommended"]) is True


def test_simulate_roi_end_to_end(tmp_path):
    scored_offers = pd.DataFrame(
        {
            "account_id": ["a1", "a2"],
            "offer_id": ["o1", "o2"],
            "offer_type": ["discount", "discount"],
            "min_value": [10, 10],
            "discount_value": [2, 2],
            "success": [1, 0],
            "y_pred_proba": [0.9, 0.01],
        }
    )
    scored_offers.to_parquet(tmp_path / "scored_offers.parquet", index=False)

    result = simulate_roi(processed_dir=str(tmp_path), margin_rate=0.3, send_cost=0.05)

    assert result["baseline_sends"] == 2
    assert result["model_sends"] == 1  # so a instancia com y_pred_proba alta compensa
    assert result["profit_lift"] == result["model_profit"] - result["baseline_profit"]
