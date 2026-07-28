"""Testes de src/models/train.py: unitario (_evaluate) e de integracao (train_response_models)."""

import numpy as np
import pandas as pd

from src.models.train import _evaluate, train_response_models


def test_evaluate_returns_expected_metric_keys():
    y_test = pd.Series([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.4, 0.5, 0.6, 0.9])

    metrics = _evaluate(y_test, proba)

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
        "brier",
    }
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def _make_model_features(n=60):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "account_id": [f"acc{i}" for i in range(n)],
            "offer_id": [f"off{i % 5}" for i in range(n)],
            "offer_type": rng.choice(["bogo", "discount", "informational"], n),
            "discount_value": rng.integers(0, 10, n),
            "min_value": rng.integers(0, 20, n),
            "duration": rng.uniform(3, 10, n),
            "channel_mobile": rng.integers(0, 2, n),
            "channel_social": rng.integers(0, 2, n),
            "channel_web": rng.integers(0, 2, n),
            "received_time": rng.uniform(0, 30, n),
            "remaining_days": rng.uniform(-5, 20, n),
            "age": rng.uniform(18, 80, n),
            "gender": rng.choice(["F", "M", "O"], n),
            "credit_card_limit": rng.uniform(30000, 120000, n),
            "is_profile_completed": rng.integers(0, 2, n),
            "registration_year": rng.integers(2013, 2019, n),
            "offers_received_prior": rng.integers(0, 5, n),
            "prior_success_rate": rng.uniform(0, 1, n),
            "prior_view_rate": rng.uniform(0, 1, n),
            "txn_count_prior": rng.integers(0, 10, n),
            "txn_amount_sum_prior": rng.uniform(0, 500, n),
            "txn_amount_avg_prior": rng.uniform(0, 50, n),
            "days_since_last_txn": rng.uniform(0, 20, n),
            "success": [i % 2 for i in range(n)],
        }
    )


def test_train_response_models_end_to_end(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    _make_model_features().to_parquet(
        processed_dir / "model_features.parquet", index=False
    )
    models_dir = tmp_path / "models"

    metrics = train_response_models(
        processed_dir=str(processed_dir), models_dir=str(models_dir)
    )

    assert set(metrics) == {"logreg", "lgbm", "xgb"}
    for model_metrics in metrics.values():
        assert {"roc_auc", "pr_auc", "brier"} <= set(model_metrics)
    assert (models_dir / "lgbm_model.joblib").exists()
    assert (processed_dir / "scored_offers.parquet").exists()
