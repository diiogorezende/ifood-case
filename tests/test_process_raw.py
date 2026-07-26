"""Testes unitarios das funcoes puras de src/data/process_raw.py."""

from src.data.process_raw import _clean_profile, _label_success


def test_clean_profile_treats_age_118_as_null(spark):
    profile = spark.createDataFrame(
        [
            (118, None, None, "acc1", "20200101"),
            (30, "M", 50000.0, "acc2", "20200101"),
        ],
        ["age", "gender", "credit_card_limit", "id", "registered_on"],
    )

    result = {row["account_id"]: row for row in _clean_profile(profile).collect()}

    assert result["acc1"]["age"] is None
    assert result["acc2"]["age"] == 30


def test_label_success_bogo_needs_view_and_completion_after_view(spark):
    df = spark.createDataFrame(
        [
            ("bogo", 1.0, 2.0, True),  # vista + completada apos a vista -> sucesso
            ("bogo", None, 2.0, True),  # sem vista -> sem sucesso
            (
                "informational",
                1.0,
                None,
                True,
            ),  # vista + transacao pos-vista -> sucesso
            ("informational", 1.0, None, False),  # vista sem transacao -> sem sucesso
        ],
        ["offer_type", "viewed_time", "completed_time", "has_post_view_transaction"],
    )

    result = [row["success"] for row in _label_success(df).collect()]

    assert result == [1, 0, 1, 0]
