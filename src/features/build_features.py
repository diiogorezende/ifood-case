"""Construcao das features finais para o modelo, a partir do dataset de
instancias de oferta e das transacoes brutas.

Espelho da logica utilizada em notebooks/06_feature_engineering.ipynb.
"""

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window


def _read_inputs(
    spark: SparkSession, processed_dir: str, raw_dir: str
) -> tuple[DataFrame, DataFrame, DataFrame]:
    offer_instances = spark.read.parquet(
        f"{processed_dir}/offer_instances_final.parquet"
    )

    # Transacoes brutas, pois vai ser necessario para o historico de compras
    transactions_raw = spark.read.option("multiline", "true").json(
        f"{raw_dir}/transactions.json"
    )
    transaction_events = transactions_raw.filter(
        F.col("event") == "transaction"
    ).select(
        "account_id",
        F.col("time_since_test_start").alias("transaction_time"),
        "value.amount",
    )
    return offer_instances, transactions_raw, transaction_events


def _add_static_features(offer_instances: DataFrame, test_end: float) -> DataFrame:
    # is_profile_completed: essa feature vai indicar se o cliente possui cadastro completo ou nao
    # registration_year: essa feature vai indicar a tenure do cliente
    # remaining_days: essa feature vai indicar quanto sobrou do periodo de teste apos o fim da janela da oferta
    return (
        offer_instances.withColumn(
            "is_profile_completed", F.col("age").isNotNull().cast("int")
        )
        .withColumn("registration_year", F.year("registered_on"))
        .withColumn("remaining_days", F.lit(test_end) - F.col("window_end"))
    )


def _encode_channels(offer_features: DataFrame) -> DataFrame:
    # Encoding de "channels": uma feature booleana para cada canal.
    # Um detalhe: "email" vai ser removido por existir em todas as ofertas
    return (
        offer_features.withColumn(
            "channel_mobile", F.array_contains("channels", "mobile").cast("int")
        )
        .withColumn(
            "channel_social", F.array_contains("channels", "social").cast("int")
        )
        .withColumn("channel_web", F.array_contains("channels", "web").cast("int"))
        .drop("channels")
    )


def _add_customer_history_features(offer_features: DataFrame) -> DataFrame:
    # A ideia vai ser olhar para a janela por "account_id"
    # Para cada instancia, olhar as ofertas anteriores do mesmo cliente
    customer_history_window = (
        Window.partitionBy("account_id")
        .orderBy("received_time")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    # offers_received_prior = numero de ofertas recebidas pelo cliente antes da oferta atual
    # prior_success_rate = taxa de sucesso do cliente em ofertas anteriores
    # prior_view_rate = taxa de visualizacao do cliente em ofertas anteriores
    return (
        offer_features.withColumn(
            "offers_received_prior", F.count("*").over(customer_history_window)
        )
        .withColumn(
            "prior_success_rate", F.avg("success").over(customer_history_window)
        )
        .withColumn(
            "prior_view_rate",
            F.avg(F.col("viewed_time").isNotNull().cast("int")).over(
                customer_history_window
            ),
        )
    )


def _build_account_timeline(
    offer_features: DataFrame, transaction_events: DataFrame
) -> DataFrame:
    # Aqui, a ideia vai ser gerar um formato comum para os datasets

    # Criar o marcador do evento de transacao, que vai ser tempo + valor da compra
    transaction_marker = transaction_events.select(
        "account_id",
        F.col("transaction_time").alias("event_time"),
        F.lit(1).alias("is_transaction"),
        "amount",
    )

    # Criar o marcador do evento de oferta recebida, que vai usar offer_id + received_time
    offer_marker = offer_features.select(
        "account_id",
        "offer_id",
        F.col("received_time").alias("event_time"),
        F.lit(0).alias("is_transaction"),
        F.lit(None).cast("double").alias("amount"),
    )

    # Agora, a ideia vai ser unir os dois markers numa linha temporal por cliente
    transaction_marker = transaction_marker.withColumn(
        "offer_id", F.lit(None).cast("string")
    )

    # Aqui sim acontece o union das duas timelines por account_id
    return transaction_marker.unionByName(offer_marker)


def _add_transaction_history_features(account_timeline: DataFrame) -> DataFrame:
    # Agora, a ideia vai ser criar uma janela agregada
    account_history_window = (
        Window.partitionBy("account_id")
        .orderBy("event_time")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    # Nessa agragacao, so contam quando is_transaction == 1, olhando somente para o historico de compras passadas do cliente
    return (
        account_timeline.withColumn(
            "txn_count_prior", F.sum("is_transaction").over(account_history_window)
        )
        .withColumn(
            "txn_amount_sum_prior",
            F.sum(F.when(F.col("is_transaction") == 1, F.col("amount"))).over(
                account_history_window
            ),
        )
        .withColumn(
            "txn_amount_avg_prior",
            F.col("txn_amount_sum_prior") / F.col("txn_count_prior"),
        )
        .withColumn(
            "last_txn_time_prior",
            F.max(F.when(F.col("is_transaction") == 1, F.col("event_time"))).over(
                account_history_window
            ),
        )
        .withColumn(
            "days_since_last_txn", F.col("event_time") - F.col("last_txn_time_prior")
        )
    )


FINAL_COLUMNS = [
    "account_id",
    "offer_id",
    "offer_type",
    "discount_value",
    "min_value",
    "duration",
    "channel_mobile",
    "channel_social",
    "channel_web",
    "received_time",
    "remaining_days",
    "age",
    "gender",
    "credit_card_limit",
    "is_profile_completed",
    "registration_year",
    "offers_received_prior",
    "prior_success_rate",
    "prior_view_rate",
    "txn_count_prior",
    "txn_amount_sum_prior",
    "txn_amount_avg_prior",
    "days_since_last_txn",
    "success",
]


def build_features(
    spark: SparkSession,
    processed_dir: str = "data/processed",
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
) -> DataFrame:
    """Constroi as features finais e grava o dataset pronto para o modelo em parquet."""
    offer_instances, transactions_raw, transaction_events = _read_inputs(
        spark, processed_dir, raw_dir
    )

    # Confirmacao do final do periodo de teste
    test_end = transactions_raw.select(F.max("time_since_test_start")).first()[0]

    offer_features = _add_static_features(offer_instances, test_end)
    offer_features = _encode_channels(offer_features)
    offer_features = _add_customer_history_features(offer_features)

    account_timeline = _build_account_timeline(offer_features, transaction_events)
    account_timeline = _add_transaction_history_features(account_timeline)

    # Extracao das features de "is_transaction"=0.
    # Com isso, mantem-se as instancias das ofertas.
    transaction_history = account_timeline.filter(F.col("is_transaction") == 0).select(
        "account_id",
        "offer_id",
        "txn_count_prior",
        "txn_amount_sum_prior",
        "txn_amount_avg_prior",
        "days_since_last_txn",
    )

    # Join por (account_id, offer_id)
    offer_features = offer_features.join(
        transaction_history, on=["account_id", "offer_id"], how="left"
    )

    # Selecao das features finais para o modelo
    model_features = offer_features.select(*FINAL_COLUMNS)

    model_features.write.mode("overwrite").parquet(
        f"{output_dir}/model_features.parquet"
    )
    return model_features


if __name__ == "__main__":
    spark_session = SparkSession.builder.appName(
        "ifood-case-build-features"
    ).getOrCreate()
    build_features(spark_session)
