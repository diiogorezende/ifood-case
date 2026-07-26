"""Simulador de ROI: usa as probabilidades preditas pelo modelo de resposta
(Parte A de notebooks/07_modeling.ipynb, persistidas em
data/processed/scored_offers.parquet) para simular o impacto financeiro de
uma politica de targeting guiada pelo modelo, comparada com a politica atual
(enviar a oferta para todo mundo).

Premissas de negocio (nao fornecidas pelos dados brutos, assumidas e
parametrizaveis via CLI):
- `margin_rate`: margem bruta do iFood sobre o gasto do cliente (`min_value`).
  Usada para converter o "gasto minimo exigido pela oferta" em receita
  incremental estimada. Default = 0.30 (30%).
- `send_cost`: custo fixo de enviar a oferta (canal/notificacao), incorrido
  independente de redencao. Default = 0.05 (unidade monetaria arbitraria).
- Nota: em todas as ofertas `bogo` deste dataset, `discount_value == min_value`
  (100% do valor minimo e devolvido ao cliente). Isso significa que `bogo` so
  e monetariamente rentavel se `margin_rate >= 1.0` (margem de 100%), o que nao
  e realista - ou seja, `bogo` tende a ser estruturalmente um "loss leader"
  (engajamento/retencao), nao uma fonte direta de lucro por redencao.
- Ofertas `informational` nao tem `discount_value`/`min_value` (sao 0 por
  definicao) e por isso nao entram na simulacao monetaria: seu valor e de
  engajamento/branding, nao de ROI direto. Ficam de fora do calculo de lucro.

A simulacao roda em cima do conjunto de teste (dados nao vistos pelo modelo
no treino), usando o `success` real de cada instancia como "o que teria
acontecido" caso a oferta fosse enviada - um backtest da politica, nao uma
projecao causal.
"""

import argparse

import pandas as pd

MONETARY_OFFER_TYPES = ["bogo", "discount"]

DEFAULT_MARGIN_RATE = 0.30
DEFAULT_SEND_COST = 0.05


def _load_scored_offers(processed_dir: str) -> pd.DataFrame:
    return pd.read_parquet(f"{processed_dir}/scored_offers.parquet")


def _add_expected_value(
    scored_offers: pd.DataFrame, margin_rate: float, send_cost: float
) -> pd.DataFrame:
    monetary_offers = scored_offers[
        scored_offers["offer_type"].isin(MONETARY_OFFER_TYPES)
    ].copy()

    # Receita incremental estimada, caso o cliente redima a oferta (gasto >= min_value).
    monetary_offers["margin_revenue"] = monetary_offers["min_value"] * margin_rate
    # Custo do desconto/recompensa, pago apenas quando a oferta e redimida.
    monetary_offers["redemption_cost"] = monetary_offers["discount_value"]

    # Valor esperado de enviar a oferta = P(sucesso) * (receita - custo de redencao) - custo fixo de envio.
    monetary_offers["expected_value"] = (
        monetary_offers["y_pred_proba"]
        * (monetary_offers["margin_revenue"] - monetary_offers["redemption_cost"])
        - send_cost
    )
    monetary_offers["send_recommended"] = monetary_offers["expected_value"] > 0

    # Lucro realizado (usando o resultado real, "success"), se a oferta fosse enviada.
    monetary_offers["realized_profit_if_sent"] = (
        monetary_offers["success"]
        * (monetary_offers["margin_revenue"] - monetary_offers["redemption_cost"])
        - send_cost
    )

    return monetary_offers


def _simulate_policies(monetary_offers: pd.DataFrame) -> dict:
    # Politica atual (baseline): enviar a oferta para todo mundo (e como os dados foram gerados).
    baseline_profit = monetary_offers["realized_profit_if_sent"].sum()
    baseline_sends = len(monetary_offers)

    # Politica guiada pelo modelo: enviar somente quando o valor esperado > 0.
    targeted = monetary_offers[monetary_offers["send_recommended"]]
    model_profit = targeted["realized_profit_if_sent"].sum()
    model_sends = len(targeted)

    return {
        "baseline_sends": baseline_sends,
        "baseline_profit": baseline_profit,
        "baseline_profit_per_send": baseline_profit / baseline_sends,
        "model_sends": model_sends,
        "model_profit": model_profit,
        "model_profit_per_send": model_profit / model_sends if model_sends else 0.0,
        "sends_avoided": baseline_sends - model_sends,
        "profit_lift": model_profit - baseline_profit,
        "profit_lift_pct": (
            (model_profit - baseline_profit) / abs(baseline_profit)
            if baseline_profit
            else float("nan")
        ),
    }


def simulate_roi(
    processed_dir: str = "data/processed",
    margin_rate: float = DEFAULT_MARGIN_RATE,
    send_cost: float = DEFAULT_SEND_COST,
) -> dict:
    """Roda a simulacao de ROI sobre o conjunto de teste em scored_offers.parquet
    e retorna um dicionario com o comparativo entre a politica atual (enviar
    para todos) e a politica guiada pelo modelo (enviar so quando o valor
    esperado e positivo).
    """
    scored_offers = _load_scored_offers(processed_dir)
    monetary_offers = _add_expected_value(scored_offers, margin_rate, send_cost)
    return _simulate_policies(monetary_offers)


def main():
    parser = argparse.ArgumentParser(
        description="Simulador de ROI de targeting de ofertas (iFood case)."
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Diretorio com scored_offers.parquet (default: data/processed).",
    )
    parser.add_argument(
        "--margin-rate",
        type=float,
        default=DEFAULT_MARGIN_RATE,
        help=f"Margem bruta sobre min_value (default: {DEFAULT_MARGIN_RATE}).",
    )
    parser.add_argument(
        "--send-cost",
        type=float,
        default=DEFAULT_SEND_COST,
        help=f"Custo fixo de envio por oferta (default: {DEFAULT_SEND_COST}).",
    )
    args = parser.parse_args()

    result = simulate_roi(args.processed_dir, args.margin_rate, args.send_cost)

    print("Simulacao de ROI - bogo/discount (conjunto de teste)")
    print(f"  premissas: margin_rate={args.margin_rate}, send_cost={args.send_cost}")
    print()
    print("  Politica atual (enviar para todos):")
    print(f"    envios: {result['baseline_sends']}")
    print(f"    lucro total: {result['baseline_profit']:.2f}")
    print(f"    lucro por envio: {result['baseline_profit_per_send']:.4f}")
    print()
    print("  Politica guiada pelo modelo (enviar so se valor esperado > 0):")
    print(f"    envios: {result['model_sends']} (evitados: {result['sends_avoided']})")
    print(f"    lucro total: {result['model_profit']:.2f}")
    print(f"    lucro por envio: {result['model_profit_per_send']:.4f}")
    print()
    print(
        f"  Ganho de lucro total com a politica do modelo: {result['profit_lift']:.2f} "
        f"({result['profit_lift_pct']:.2%})"
    )


if __name__ == "__main__":
    main()
