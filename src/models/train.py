"""Treinamento dos modelos de resposta/propensao e do T-Learner de
uplift.

Espelho da logica utilizada em notebooks/07_modeling.ipynb.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ID_COLS = ["account_id", "offer_id"]
TARGET_COL = "success"

SHARED_COLS = [
    "age",
    "gender",
    "credit_card_limit",
    "is_profile_completed",
    "registration_year",
    "txn_count_prior",
    "txn_amount_sum_prior",
    "txn_amount_avg_prior",
    "days_since_last_txn",
]


# ---------------------------------------------------------------------------
# Parte A - modelo de resposta/propensao
# ---------------------------------------------------------------------------


def _read_model_features(processed_dir: str) -> pd.DataFrame:
    return pd.read_parquet(f"{processed_dir}/model_features.parquet")


def _prepare_response_data(
    model_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Index]:
    feature_cols = [
        c for c in model_features.columns if c not in ID_COLS + [TARGET_COL]
    ]

    X = model_features[feature_cols].copy()
    y = model_features[TARGET_COL].copy()

    categorical_cols = X.select_dtypes(include="str").columns
    X[categorical_cols] = X[categorical_cols].astype("category")

    return X, y, feature_cols, categorical_cols


def _train_logreg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: pd.Index,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    logreg_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    logreg_pipeline.fit(X_train, y_train)
    return logreg_pipeline


def _train_lgbm(
    X_train: pd.DataFrame, y_train: pd.Series, categorical_cols: pd.Index
) -> LGBMClassifier:
    lgbm_model = LGBMClassifier(random_state=42)
    lgbm_model.fit(X_train, y_train, categorical_feature=list(categorical_cols))
    return lgbm_model


def _train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    xgb_model = XGBClassifier(
        enable_categorical=True, tree_method="hist", random_state=42
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model


def _evaluate(y_test: pd.Series, proba) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
    }


def train_response_models(
    processed_dir: str = "data/processed",
    models_dir: str = "src/models",
) -> dict[str, dict[str, float]]:
    """Treina os 3 modelos de resposta (LogReg, LightGBM, XGBoost), persiste o
    modelo final (LightGBM) e retorna as metricas de cada um no conjunto de teste.
    """
    model_features = _read_model_features(processed_dir)
    X, y, feature_cols, categorical_cols = _prepare_response_data(model_features)
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logreg_pipeline = _train_logreg(X_train, y_train, numeric_cols, categorical_cols)
    lgbm_model = _train_lgbm(X_train, y_train, categorical_cols)
    xgb_model = _train_xgb(X_train, y_train)

    metrics = {
        "logreg": _evaluate(y_test, logreg_pipeline.predict_proba(X_test)[:, 1]),
        "lgbm": _evaluate(y_test, lgbm_model.predict_proba(X_test)[:, 1]),
        "xgb": _evaluate(y_test, xgb_model.predict_proba(X_test)[:, 1]),
    }

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(logreg_pipeline, models_path / "logreg_model.joblib")
    joblib.dump(lgbm_model, models_path / "lgbm_model.joblib")
    joblib.dump(xgb_model, models_path / "xgb_model.joblib")

    metadata = {
        "feature_cols": feature_cols,
        "categorical_cols": list(categorical_cols),
        "target_col": TARGET_COL,
        "id_cols": ID_COLS,
        "final_model": "lgbm_model.joblib",
    }
    with open(models_path / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
    scored_offers = model_features.loc[
        X_test.index,
        ID_COLS + ["offer_type", "discount_value", "min_value", TARGET_COL],
    ].copy()
    scored_offers["y_pred_proba"] = lgbm_proba
    scored_offers.to_parquet(f"{processed_dir}/scored_offers.parquet", index=False)

    return metrics


# ---------------------------------------------------------------------------
# Parte B - T-Learner de uplift (stretch goal)
# ---------------------------------------------------------------------------


def _read_t_learner_inputs(
    processed_dir: str, raw_dir: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    offer_instances = pd.read_parquet(f"{processed_dir}/offer_instances_final.parquet")
    transactions_raw = pd.read_json(f"{raw_dir}/transactions.json")

    transaction_events = transactions_raw.loc[
        transactions_raw["event"] == "transaction",
        ["account_id", "time_since_test_start"],
    ].rename(columns={"time_since_test_start": "transaction_time"})

    test_end = transactions_raw["time_since_test_start"].max()

    return offer_instances, transactions_raw, transaction_events, test_end


def _build_organic_windows(
    offer_instances: pd.DataFrame,
    transaction_events: pd.DataFrame,
    test_end: float,
) -> pd.DataFrame:
    offer_instances_sorted = offer_instances.sort_values(
        ["account_id", "received_time"]
    )
    prev_window_end = offer_instances_sorted.groupby("account_id")["window_end"].shift(
        1
    )
    gaps_before = pd.DataFrame(
        {
            "account_id": offer_instances_sorted["account_id"],
            "gap_start": prev_window_end.fillna(0.0),
            "gap_end": offer_instances_sorted["received_time"],
        }
    )

    last_window_end = offer_instances.groupby("account_id")["window_end"].max()
    gaps_after = pd.DataFrame(
        {
            "account_id": last_window_end.index,
            "gap_start": last_window_end.values,
            "gap_end": test_end,
        }
    )

    all_gaps = pd.concat([gaps_before, gaps_after], ignore_index=True)
    all_gaps["gap_length"] = all_gaps["gap_end"] - all_gaps["gap_start"]

    organic_windows = all_gaps[all_gaps["gap_length"] >= 3].reset_index(drop=True)
    organic_windows["gap_id"] = organic_windows.index

    merged = organic_windows.merge(transaction_events, on="account_id", how="left")
    in_window = (merged["transaction_time"] >= merged["gap_start"]) & (
        merged["transaction_time"] < merged["gap_end"]
    )
    has_transaction_by_gap = (
        merged.assign(in_window=in_window).groupby("gap_id")["in_window"].any()
    )
    organic_windows["has_organic_transaction"] = (
        organic_windows["gap_id"].map(has_transaction_by_gap).fillna(False)
    )

    return organic_windows


def _build_demographics(offer_instances: pd.DataFrame) -> pd.DataFrame:
    demographics = (
        offer_instances[
            ["account_id", "age", "gender", "credit_card_limit", "registered_on"]
        ]
        .drop_duplicates(subset="account_id")
        .copy()
    )
    demographics["is_profile_completed"] = demographics["age"].notna().astype(int)
    demographics["registration_year"] = pd.to_datetime(
        demographics["registered_on"]
    ).dt.year
    return demographics


def _build_transaction_history(transactions_raw: pd.DataFrame) -> pd.DataFrame:
    transaction_history = transactions_raw.loc[
        transactions_raw["event"] == "transaction",
        ["account_id", "time_since_test_start", "value"],
    ].copy()
    transaction_history["amount"] = transaction_history["value"].apply(
        lambda v: v["amount"]
    )
    transaction_history = (
        transaction_history.rename(
            columns={"time_since_test_start": "transaction_time"}
        )
        .drop(columns="value")
        .sort_values(["account_id", "transaction_time"])
    )

    transaction_history["txn_count_prior"] = (
        transaction_history.groupby("account_id").cumcount() + 1
    )
    transaction_history["txn_amount_sum_prior"] = transaction_history.groupby(
        "account_id"
    )["amount"].cumsum()
    transaction_history["last_txn_time_prior"] = transaction_history["transaction_time"]

    return transaction_history


def _build_control_features(
    organic_windows: pd.DataFrame,
    transaction_history: pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    control_features = pd.merge_asof(
        organic_windows.sort_values("gap_start"),
        transaction_history[
            [
                "account_id",
                "transaction_time",
                "txn_count_prior",
                "txn_amount_sum_prior",
                "last_txn_time_prior",
            ]
        ].sort_values("transaction_time"),
        left_on="gap_start",
        right_on="transaction_time",
        by="account_id",
        direction="backward",
    )

    control_features["txn_count_prior"] = control_features["txn_count_prior"].fillna(0)
    control_features["txn_amount_sum_prior"] = control_features[
        "txn_amount_sum_prior"
    ].fillna(0.0)
    control_features["txn_amount_avg_prior"] = (
        control_features["txn_amount_sum_prior"] / control_features["txn_count_prior"]
    )
    control_features["days_since_last_txn"] = (
        control_features["gap_start"] - control_features["last_txn_time_prior"]
    )

    return control_features.merge(demographics, on="account_id", how="left")


def _align_features(
    model_features: pd.DataFrame, control_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    treated_features = model_features[
        ["account_id", "offer_id", "offer_type"] + SHARED_COLS
    ].copy()

    control_features_final = control_features[
        ["account_id", "gap_id"] + SHARED_COLS
    ].copy()
    control_features_final["is_profile_completed"] = control_features_final[
        "is_profile_completed"
    ].astype("int32")

    return treated_features, control_features_final


def _train_t_learner_models(
    treated_features: pd.DataFrame,
    treated_labels,
    control_features_final: pd.DataFrame,
    control_labels,
) -> tuple[LGBMClassifier, LGBMClassifier]:
    X_control = control_features_final[SHARED_COLS].copy()
    X_control["gender"] = X_control["gender"].astype("category")
    y_control = pd.Series(control_labels, name="y")

    Xc_train, _, yc_train, _ = train_test_split(
        X_control, y_control, test_size=0.2, random_state=42, stratify=y_control
    )
    control_model = LGBMClassifier(random_state=42)
    control_model.fit(Xc_train, yc_train, categorical_feature=["gender"])

    X_treated = treated_features[SHARED_COLS + ["offer_type"]].copy()
    X_treated[["gender", "offer_type"]] = X_treated[["gender", "offer_type"]].astype(
        "category"
    )
    y_treated = pd.Series(treated_labels, name="y")

    Xt_train, _, yt_train, _ = train_test_split(
        X_treated, y_treated, test_size=0.2, random_state=42, stratify=y_treated
    )
    treated_model = LGBMClassifier(random_state=42)
    treated_model.fit(Xt_train, yt_train, categorical_feature=["gender", "offer_type"])

    return control_model, treated_model


def _compute_uplift_by_offer_type(
    treated_model: LGBMClassifier,
    control_model: LGBMClassifier,
    treated_features: pd.DataFrame,
) -> pd.Series:
    X_treated = treated_features[SHARED_COLS + ["offer_type"]].copy()
    X_treated[["gender", "offer_type"]] = X_treated[["gender", "offer_type"]].astype(
        "category"
    )
    p_treated = treated_model.predict_proba(X_treated)[:, 1]

    X_counterfactual = treated_features[SHARED_COLS].copy()
    X_counterfactual["gender"] = X_counterfactual["gender"].astype("category")
    p_control = control_model.predict_proba(X_counterfactual)[:, 1]

    uplift = p_treated - p_control

    return (
        pd.DataFrame({"offer_type": treated_features["offer_type"], "uplift": uplift})
        .groupby("offer_type")["uplift"]
        .mean()
        .sort_values(ascending=False)
    )


def train_t_learner(
    processed_dir: str = "data/processed",
    raw_dir: str = "data/raw",
    models_dir: str = "src/models",
) -> pd.Series:
    """Reconstroi as janelas organicas (pseudo-controle), treina os dois
    classificadores do T-Learner, persiste os modelos e retorna o uplift
    estimado por offer_type.
    """
    model_features = _read_model_features(processed_dir)
    offer_instances, transactions_raw, transaction_events, test_end = (
        _read_t_learner_inputs(processed_dir, raw_dir)
    )

    organic_windows = _build_organic_windows(
        offer_instances, transaction_events, test_end
    )
    demographics = _build_demographics(offer_instances)
    transaction_history = _build_transaction_history(transactions_raw)
    control_features = _build_control_features(
        organic_windows, transaction_history, demographics
    )

    treated_features, control_features_final = _align_features(
        model_features, control_features
    )

    control_labels = control_features["has_organic_transaction"].astype(int).values
    treated_labels = model_features["success"].astype(int).values

    control_model, treated_model = _train_t_learner_models(
        treated_features, treated_labels, control_features_final, control_labels
    )

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(control_model, models_path / "t_learner_control_model.joblib")
    joblib.dump(treated_model, models_path / "t_learner_treated_model.joblib")

    return _compute_uplift_by_offer_type(treated_model, control_model, treated_features)


if __name__ == "__main__":
    response_metrics = train_response_models()
    print("Metricas dos modelos de resposta:")
    for model_name, model_metrics in response_metrics.items():
        print(f"  {model_name}: {model_metrics}")

    uplift_by_offer_type = train_t_learner()
    print("Uplift estimado por offer_type (T-Learner):")
    print(uplift_by_offer_type)
