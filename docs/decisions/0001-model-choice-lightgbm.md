# ADR 0001: LightGBM with cost-aware decisions

Status: accepted

## Context
The input is a tabular incident table with about 75 numeric features: counts, shares, durations, and encoded alert and detector identifiers. There are three classes with moderate imbalance (roughly 41% BenignPositive, 37% TruePositive, 22% FalsePositive in the sample). Missing a real threat costs far more than escalating a harmless incident.

## Decision
- Model: LightGBM multiclass, with early stopping on the validation split.
- Baselines: majority class, and logistic regression on log-scaled features.
- No class weights. The model keeps natural probabilities, and the class asymmetry is handled by a cost matrix at decision time.
- Decision rule: pick the class with the lowest expected cost, and report the plain argmax rule next to it.
- High-cardinality identifiers (dominant alert title and detector) are target-encoded out-of-fold, fit on the training split only.
- Model choice and early stopping use the validation split only. The test split is used for reporting.

## Alternatives considered
- XGBoost: similar accuracy on tabular data, but LightGBM trains faster on millions of rows.
- Random forest: slower, usually weaker on this kind of data.
- Neural networks: no clear gain on tabular data, and harder to explain.
- Class weights: would distort the probabilities that the cost rule depends on.

## Consequences
- The cost matrix values are illustrative assumptions, not measured costs. Results should be shown across a range of assumptions.
- Probabilities are used directly in the cost rule, so their calibration should be checked before relying on the cost numbers.
- Macro-F1 and cost per incident can disagree. Both are reported.