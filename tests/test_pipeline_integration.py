"""Teste de integracao: raw (json) -> process_raw_data -> build_features."""

import json

from src.data.process_raw import process_raw_data
from src.features.build_features import build_features


def _write_raw_fixture(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "offers.json").write_text(
        json.dumps(
            [
                {
                    "id": "off1",
                    "offer_type": "bogo",
                    "channels": ["web"],
                    "discount_value": 5,
                    "duration": 5.0,
                    "min_value": 5,
                }
            ]
        )
    )
    (raw_dir / "profile.json").write_text(
        json.dumps(
            [
                {
                    "id": "acc1",
                    "age": 30,
                    "gender": "M",
                    "credit_card_limit": 50000.0,
                    "registered_on": "20200101",
                }
            ]
        )
    )
    (raw_dir / "transactions.json").write_text(
        json.dumps(
            [
                {
                    "event": "offer received",
                    "account_id": "acc1",
                    "time_since_test_start": 0.0,
                    "value": {"offer id": "off1"},
                },
                {
                    "event": "offer viewed",
                    "account_id": "acc1",
                    "time_since_test_start": 1.0,
                    "value": {"offer id": "off1"},
                },
                {
                    "event": "offer completed",
                    "account_id": "acc1",
                    "time_since_test_start": 2.0,
                    "value": {"offer_id": "off1", "reward": 5},
                },
                {
                    "event": "transaction",
                    "account_id": "acc1",
                    "time_since_test_start": 3.0,
                    "value": {"amount": 10.0},
                },
            ]
        )
    )


def test_process_raw_and_build_features_pipeline(spark, tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_fixture(raw_dir)
    processed_dir = tmp_path / "processed"

    offer_instances = process_raw_data(
        spark, raw_dir=str(raw_dir), output_dir=str(processed_dir)
    ).collect()

    assert len(offer_instances) == 1
    assert offer_instances[0]["success"] == 1  # vista + completada apos a vista

    model_features = build_features(
        spark,
        processed_dir=str(processed_dir),
        raw_dir=str(raw_dir),
        output_dir=str(processed_dir),
    ).collect()

    assert len(model_features) == 1
    row = model_features[0]
    assert row["channel_web"] == 1
    assert "channels" not in row.asDict()
    # unica oferta do cliente -> sem historico anterior (guarda contra vazamento temporal)
    assert row["offers_received_prior"] == 0
    assert row["txn_count_prior"] is None
