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
  inviável para uplift clássico. Proposta: T-Learner
  com pseudo-controle (janelas "orgânicas" sem oferta ativa).
- **Vazamento temporal**: todas as features de histórico (RFM de transações, taxa de sucesso/
  visualização anterior) usam apenas eventos *estritamente anteriores* ao recebimento da oferta
  (`rowsBetween(unboundedPreceding, -1)`).
- **Feature `email` (canal)** removida (constante, 100% das ofertas, zero variância).

## Resultados — modelo de resposta

3 modelos comparados no conjunto de teste (20%, holdout estratificado); **LightGBM** escolhido
como final:

| Modelo    | Accuracy* | Precision* | Recall* | ROC-AUC | PR-AUC | Brier |
|-----------|-----------|------------|---------|---------|--------|-------|
| LogReg    | 0.688 | 0.626 | 0.491 | 0.738 | 0.627 | 0.198 |
| XGBoost   | 0.732 | 0.670 | 0.613 | 0.799 | 0.701 | 0.176 |
| **LightGBM** | **0.739** | **0.684** | **0.613** | **0.807** | **0.712** | **0.173** |

\* Accuracy/Precision/Recall calculados no corte arbitrário de probabilidade > 0.5 (não
otimizado para custo de negócio); usados apenas para leitura complementar, o ranking entre
modelos segue as métricas independentes de threshold (ROC-AUC/PR-AUC/Brier). Como referência,
o baseline "sempre prever não-sucesso" já teria accuracy ≈ 61% (taxa de sucesso ≈ 38.93%).

(Métricas completas de todos os modelos em `src/models/model_metadata.json` / notebook 07.)

- É necessário, como próximo passo, escolher o threshold de decisão via otimização de valor esperado, em vez de usar o corte fixo de 0.5 nas métricas de accuracy/precision/ recall reportadas no notebook 07.

## Resultados — T-Learner

Uplift médio `(P(sucesso tratado) − P(compra orgânica no controle))` por tipo de oferta, todos
**negativos** (discount −0.22, informational −0.25, bogo −0.28). Resultado tratado como
**não conclusivo**, não como efeito causal real: a definição de sucesso é assimétrica, pois
tratado exige visualização + redenção/transação e controle exige apenas qualquer transação. E
as janelas de controle (gaps sem oferta) tendem a ser mais longas que a duração típica da
oferta, ou seja, ambos inflam artificialmente a taxa do controle.

**Checagem com métrica comparável**: usando a mesma janela orgânica, mas uma métrica comparável
(qualquer transação, sem exigir visualização) em vez de `success`, o notebook 05 recalcula o
uplift tratado vs. orgânico (62.71%) e encontra diferenças pequenas ou negativas para todos os
tipos: `bogo` +1.8 p.p., `discount` −3.1 p.p. e `informational` −24.0 p.p.. Ou seja, mesmo sob a métrica mais
permissiva, nenhum `offer_type` mostra incremento real sobre o comportamento orgânico,
o que é consistente com o sinal negativo do T-Learner. Informa que o uplift não deve ser considerado de forma isolada.

## Simulação de ROI (`main.py`)

Backtest no conjunto de teste, usando o `success` real como proxy do resultado caso a oferta
fosse enviada. Só `bogo`/`discount` entram na conta monetária (`informational` não tem
`min_value`/`discount_value`).

- Todos os `bogo` têm `discount_value == min_value` (devolução de 100%) → só é lucrativo com
  margem ≥100%, ou seja, é estruturalmente um *loss-leader* de engajamento, não de lucro.
- Com premissas default (`margin_rate=0.30`, `send_cost=0.05`): política atual (enviar para
  todos, 10135 envios) resulta em **-9152** de lucro; política guiada pelo modelo (enviar só
  quando valor esperado > 0, acarreta em 3428 envios, 6707 evitados) resulta em **+1246** de lucro
  (**+113.6%** de lift). Parâmetros ajustáveis via `--margin-rate`/`--send-cost`.

## Pontos fortes

- Pipeline ponta-a-ponta reprodutível (raw → decisão de negócio), com notebooks de exploração
  e módulos `src/` espelhando fielmente a lógica final, validados numericamente contra os
  notebooks.
- Cuidado explícito com vazamento temporal (features de histórico só usam o passado).
- Discussão aberta de premissas de negócio (margem, custo de envio) e como elas mudam a
  decisão ótima.
- Simulação de ROI compara diretamente política atual vs. guiada pelo modelo, em unidades de
  negócio, não só métricas de classificação.

## Fragilidades e limitações

- **Uplift não é causal**: pseudo-controle tem viés estrutural conhecido (assimetria
  de outcome e de duração de janela); resultado não deve orientar decisão de negócio sozinho.
- **ROI depende de premissas não observadas nos dados** (`margin_rate`, `send_cost`), números
  absolutos são sensíveis a esses parâmetros, só as ordens de grandeza/heterogeneidade são
  robustas.
- **Sem validação temporal** — split é aleatório estratificado, não por tempo;
  em produção o modelo será usado para prever o futuro a partir do passado.
- Ambiguidade não resolvida nas tags de tempo empatadas (transação e oferta no mesmo timestamp)
  no PySpark window; edge case raro, aceito como limitação.

## Próximos passos

- Validação temporal (split out-of-time) e recalibração periódica do modelo.
- Uplift causal mais robusto (matching/propensity score, ou desenho experimental real com
  grupo de controle randomizado) em vez do pseudo-controle atual.
- Tuning de hiperparâmetros + calibração de probabilidade para o simulador de
  ROI ser mais sensível a limiares de decisão.
- Otimização por cliente/oferta (não só threshold global de valor esperado) e teste A/B online
  para validar o lift estimado em produção.

## Como rodar

```bash
uv run python -m src.data.process_raw       # raw -> offer_instances_final.parquet
uv run python -m src.features.build_features  # -> model_features.parquet
uv run python -m src.models.train           # treina modelos + scored_offers.parquet
uv run python main.py                       # simulação de ROI (--margin-rate, --send-cost)
```

## Como rodar os testes

```bash
uv run python -m pytest tests/ -v
```

Testes unitários (funções puras de `process_raw.py`/`train.py`/`main.py`) e de integração
(pipeline `process_raw_data` → `build_features`, e `train_response_models`/`simulate_roi`
ponta-a-ponta com dados sintéticos em `tmp_path`), em `tests/`.
