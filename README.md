# iFood Case — Targeting de Ofertas e Simulação de ROI

Pipeline completo (EDA → processamento → features → modelagem → simulação de ROI) para decidir
**para quem enviar cada oferta** do iFood, a partir de `offers.json`, `profile.json` e
`transactions.json` (~17k clientes, ~306k eventos, ~63k instâncias de oferta válidas).

## Pipeline

```
data/raw/*.json → src/data/process_raw.py → data/processed/offer_instances_final.parquet
                → src/features/build_features.py → data/processed/model_features.parquet
                → src/models/train.py → src/models/*.joblib + scored_offers.parquet
                → main.py (simulação de ROI)
```

Notebooks (`notebooks/01`–`07`) documentam a exploração e as decisões; `src/` espelha a versão
final e reprodutível de cada etapa (PySpark no processamento/features, pandas na modelagem).

## Principais decisões e achados

- **Label (`success`)**: exige visualização prévia da oferta. `bogo`/`discount` = vista +
  completada (redenção); `informational` (sem evento de redenção) = vista + transação
  subsequente. Taxas de sucesso: discount 41.6%, informational 38.7%, bogo 36.3%.
- **Sem grupo de controle nativo**: só 6 de 17k clientes nunca receberam oferta →
  inviável para uplift clássico. Solução: response model como entrega principal; T-Learner
  com pseudo-controle (janelas "orgânicas" sem oferta ativa) como stretch goal.
- **Vazamento temporal**: todas as features de histórico (RFM de transações, taxa de sucesso/
  visualização anterior) usam apenas eventos *estritamente anteriores* ao recebimento da oferta
  (`rowsBetween(unboundedPreceding, -1)`).
- **Feature `email` (canal)** removida (constante, 100% das ofertas, zero variância).

## Resultados — modelo de resposta (Parte A)

3 modelos comparados no conjunto de teste (20%, holdout estratificado); **LightGBM** escolhido
como final:

| Modelo    | ROC-AUC | PR-AUC | Brier |
|-----------|---------|--------|-------|
| LogReg    | baseline linear | — | — |
| XGBoost   | próximo ao LightGBM | — | — |
| **LightGBM** | **0.807** | **0.712** | **0.173** |

(Métricas completas de todos os modelos em `src/models/model_metadata.json` / notebook 07.)

## Resultados — uplift T-Learner (Parte B, stretch goal)

Uplift médio (P(sucesso tratado) − P(compra orgânica no controle)) por tipo de oferta, todos
**negativos** (discount −0.22, informational −0.25, bogo −0.28). Resultado tratado como
**não conclusivo**, não como efeito causal real: a definição de sucesso é assimétrica
(tratado exige visualização + redenção/transação; controle exige apenas qualquer transação) e
as janelas de controle (gaps sem oferta) tendem a ser mais longas que a duração típica da
oferta — ambos inflam artificialmente a taxa do controle.

## Simulação de ROI (`main.py`)

Backtest no conjunto de teste, usando o `success` real como proxy do resultado caso a oferta
fosse enviada. Só `bogo`/`discount` entram na conta monetária (`informational` não tem
`min_value`/`discount_value` — é lever de engajamento, não de receita direta).

- Todos os `bogo` têm `discount_value == min_value` (devolução de 100%) → só é lucrativo com
  margem ≥100%, ou seja, é estruturalmente um *loss-leader* de engajamento, não de lucro.
- Com premissas default (`margin_rate=0.30`, `send_cost=0.05`): política atual (enviar para
  todos, 10 135 envios) resulta em **-9 152** de lucro; política guiada pelo modelo (enviar só
  quando valor esperado > 0, 3 428 envios, 6 707 evitados) resulta em **+1 246** de lucro
  (**+113.6%** de lift). Parâmetros ajustáveis via `--margin-rate`/`--send-cost`.

## Pontos fortes

- Pipeline ponta-a-ponta reprodutível (raw → decisão de negócio), com notebooks de exploração
  e módulos `src/` espelhando fielmente a lógica final, validados numericamente contra os
  notebooks.
- Cuidado explícito com vazamento temporal (features de histórico só usam o passado).
- Discussão honesta de premissas de negócio (margem, custo de envio) e como elas mudam a
  decisão ótima (calibração do breakeven por tipo de oferta).
- Simulação de ROI compara diretamente política atual vs. guiada pelo modelo, em unidades de
  negócio (lucro), não só métricas de classificação.

## Fragilidades e limitações

- **Uplift (Parte B) não é causal**: pseudo-controle tem viés estrutural conhecido (assimetria
  de outcome e de duração de janela); resultado não deve orientar decisão de negócio sozinho.
- **ROI depende de premissas não observadas nos dados** (`margin_rate`, `send_cost`) — números
  absolutos são sensíveis a esses parâmetros, só as ordens de grandeza/heterogeneidade são
  robustas.
- **Sem tuning de hiperparâmetros** — modelos usam parâmetros default do scikit-learn/LightGBM/
  XGBoost; sem busca de espaço de hiperparâmetros nem calibração de probabilidade.
- **Sem validação temporal (walk-forward)** — split é aleatório estratificado, não por tempo;
  em produção o modelo será usado para prever o futuro a partir do passado.
- Ambiguidade não resolvida nas tags de tempo empatadas (transação e oferta no mesmo timestamp)
  no PySpark window; edge case raro, aceito como limitação.

## Próximos passos

- Validação temporal (split out-of-time) e recalibração periódica do modelo.
- Uplift causal mais robusto (matching/propensity score, ou desenho experimental real com
  grupo de controle randomizado) em vez do pseudo-controle atual.
- Tuning de hiperparâmetros + calibração de probabilidade (Platt/isotonic) para o simulador de
  ROI ser mais sensível a limiares de decisão.
- Otimização por cliente/oferta (não só threshold global de valor esperado) e teste A/B online
  para validar o lift estimado em produção.
- Deck de apresentação executiva (`presentation/`) com os achados acima.

## Como rodar

```bash
uv run python -m src.data.process_raw       # raw -> offer_instances_final.parquet
uv run python -m src.features.build_features  # -> model_features.parquet
uv run python -m src.models.train           # treina modelos + scored_offers.parquet
uv run python main.py                       # simulação de ROI (--margin-rate, --send-cost)
```
