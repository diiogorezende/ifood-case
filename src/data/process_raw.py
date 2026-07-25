"""Processamento dos dados brutos (offers, profile, transactions) em um
dataset unificado de instancias de oferta, pronto para feature engineering
e modelagem.

Espelho da logica utilizada em notebooks/04_processing.ipynb.
"""

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window


def _read_raw_data(
    spark: SparkSession, raw_dir: str
) -> tuple[DataFrame, DataFrame, DataFrame]:
    offers = spark.read.option("multiline", "true").json(f"{raw_dir}/offers.json")
    profile = spark.read.option("multiline", "true").json(f"{raw_dir}/profile.json")
    transactions = spark.read.option("multiline", "true").json(
        f"{raw_dir}/transactions.json"
    )
    return offers, profile, transactions


def _clean_profile(profile: DataFrame) -> DataFrame:
    # age=118 e um sentinel para cadastro incompleto (bate com nulos de gender/credit_card_limit).
    profile_clean = profile.withColumn(
        "age", F.when(F.col("age") == 118, None).otherwise(F.col("age"))
    )
    profile_clean = profile_clean.withColumn(
        "registered_on", F.to_date(F.col("registered_on"), "yyyyMMdd")
    )
    return profile_clean.withColumnRenamed("id", "account_id")


def _clean_transactions(transactions: DataFrame) -> DataFrame:
    # Padroniza as duas escritas de offer id ("offer id" vs "offer_id") e extrai amount/reward do struct "value".
    return (
        transactions.withColumn(
            "offer_id", F.coalesce(F.col("value.`offer id`"), F.col("value.offer_id"))
        )
        .withColumn("amount", F.col("value.amount"))
        .withColumn("reward", F.col("value.reward"))
        .drop("value")
    )


def _dedup_offer_received(transactions_clean: DataFrame) -> DataFrame:
    # Mantem apenas a primeira ocorrencia de "offer received" por (account_id, offer_id).
    window = Window.partitionBy("account_id", "offer_id").orderBy(
        "time_since_test_start"
    )
    return (
        transactions_clean.filter(F.col("event") == "offer received")
        .withColumn("row_number", F.row_number().over(window))
        .filter(F.col("row_number") == 1)
        .drop("row_number")
    )


def _build_offer_instances(
    offer_received_dedup: DataFrame, offers_clean: DataFrame
) -> DataFrame:
    # Cada "offer received" e o inicio de uma instancia de oferta; junta metadados e calcula o fim da janela.
    return (
        offer_received_dedup.withColumnRenamed("time_since_test_start", "received_time")
        .drop("event", "amount", "reward")
        .join(offers_clean, on="offer_id", how="left")
        .withColumn("window_end", F.col("received_time") + F.col("duration"))
    )


def _attach_first_event_in_window(
    offer_instances: DataFrame,
    events: DataFrame,
    time_col: str,
    left_alias: str,
    right_alias: str,
) -> DataFrame:
    # Padrao reutilizado: range-join (condicao de tempo dentro do join, com aliases para evitar
    # o self-reference bug quando os DataFrames compartilham lineage) + dedup pela primeira ocorrencia.
    left = offer_instances.alias(left_alias)
    right = events.alias(right_alias)

    joined = (
        left.join(
            right,
            on=(
                (
                    F.col(f"{left_alias}.account_id")
                    == F.col(f"{right_alias}.account_id")
                )
                & (F.col(f"{left_alias}.offer_id") == F.col(f"{right_alias}.offer_id"))
                & (
                    F.col(f"{right_alias}.{time_col}")
                    >= F.col(f"{left_alias}.received_time")
                )
                & (
                    F.col(f"{right_alias}.{time_col}")
                    <= F.col(f"{left_alias}.window_end")
                )
            ),
            how="left",
        )
        .drop(F.col(f"{right_alias}.account_id"))
        .drop(F.col(f"{right_alias}.offer_id"))
    )

    dedup_window = Window.partitionBy("account_id", "offer_id").orderBy(
        F.col(time_col).asc_nulls_last()
    )
    return (
        joined.withColumn("row_number", F.row_number().over(dedup_window))
        .filter(F.col("row_number") == 1)
        .drop("row_number")
    )


def _attach_has_post_view_transaction(
    offer_instances_completed: DataFrame, transaction_events: DataFrame
) -> DataFrame:
    # "transaction" nao tem offer_id; o match e por account_id, exigindo transaction_time apos o viewed_time.
    oic = offer_instances_completed.alias("oic")
    te = transaction_events.alias("te")

    instances_with_post_view_transaction = oic.join(
        te,
        on=(
            (F.col("oic.account_id") == F.col("te.account_id"))
            & (F.col("te.transaction_time") >= F.col("oic.viewed_time"))
            & (F.col("te.transaction_time") <= F.col("oic.window_end"))
        ),
        how="left_semi",
    ).select("account_id", "offer_id")

    return offer_instances_completed.join(
        instances_with_post_view_transaction.withColumn(
            "has_post_view_transaction", F.lit(True)
        ),
        on=["account_id", "offer_id"],
        how="left",
    ).fillna(False, subset=["has_post_view_transaction"])


def _label_success(offer_instances_completed: DataFrame) -> DataFrame:
    # Opcao B: exige visualizacao previa, para garantir que a conversao foi influenciada pela oferta.
    # bogo/discount: vista E completada dentro da janela, com conclusao apos a visualizacao.
    # informational: nao existe "offer completed"; sucesso = vista E transacao pos-view.
    return offer_instances_completed.withColumn(
        "success",
        F.when(
            F.col("offer_type").isin("bogo", "discount"),
            F.col("viewed_time").isNotNull()
            & F.col("completed_time").isNotNull()
            & (F.col("completed_time") >= F.col("viewed_time")),
        )
        .when(
            F.col("offer_type") == "informational",
            F.col("viewed_time").isNotNull() & F.col("has_post_view_transaction"),
        )
        .otherwise(False)
        .cast("int"),
    )


FINAL_COLUMNS = [
    "account_id",
    "offer_id",
    "offer_type",
    "channels",
    "discount_value",
    "min_value",
    "duration",
    "received_time",
    "viewed_time",
    "completed_time",
    "window_end",
    "reward",
    "has_post_view_transaction",
    "success",
    "age",
    "gender",
    "credit_card_limit",
    "registered_on",
]


def process_raw_data(
    spark: SparkSession,
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
) -> DataFrame:
    """Processa os dados brutos e grava o dataset final de instancias de oferta em parquet."""
    offers, profile, transactions = _read_raw_data(spark, raw_dir)

    offers_clean = offers.withColumnRenamed("id", "offer_id")
    profile_clean = _clean_profile(profile)
    transactions_clean = _clean_transactions(transactions)

    offer_received_dedup = _dedup_offer_received(transactions_clean)
    offer_instances = _build_offer_instances(offer_received_dedup, offers_clean)

    viewed_events = transactions_clean.filter(F.col("event") == "offer viewed").select(
        "account_id", "offer_id", F.col("time_since_test_start").alias("viewed_time")
    )
    offer_instances_viewed = _attach_first_event_in_window(
        offer_instances, viewed_events, "viewed_time", "oi", "ve"
    )

    completed_events = transactions_clean.filter(
        F.col("event") == "offer completed"
    ).select(
        "account_id",
        "offer_id",
        F.col("time_since_test_start").alias("completed_time"),
        "reward",
    )
    offer_instances_completed = _attach_first_event_in_window(
        offer_instances_viewed, completed_events, "completed_time", "oiv", "ce"
    )

    transaction_events = transactions_clean.filter(
        F.col("event") == "transaction"
    ).select(
        "account_id", F.col("time_since_test_start").alias("transaction_time"), "amount"
    )
    offer_instances_completed = _attach_has_post_view_transaction(
        offer_instances_completed, transaction_events
    )

    offer_instances_labeled = _label_success(offer_instances_completed)

    offer_instances_final = offer_instances_labeled.join(
        profile_clean, on="account_id", how="left"
    ).select(*FINAL_COLUMNS)

    offer_instances_final.write.mode("overwrite").parquet(
        f"{output_dir}/offer_instances_final.parquet"
    )
    return offer_instances_final


if __name__ == "__main__":
    spark_session = SparkSession.builder.appName("ifood-case-process-raw").getOrCreate()
    process_raw_data(spark_session)
