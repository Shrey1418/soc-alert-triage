# ADR 0001: LightGBM with cost-aware decisions



## Decision
- Model: LightGBM multiclass, with early stopping on the validation split.
- Baselines: majority class, and logistic regression on log-scaled features.
- No class weights. The model keeps natural probabilities, and the cost asymmetry is handled by a cost matrix at decision time.
- Decision rule: pick the class with the lowest expected cost, and report the plain argmax rule next to it.
- Dominant alert title and detector are target-encoded out of fold, fitted on the training split only.
- Model choice and early stopping use the validation split only. The test split is used for reporting.

## Evidence on the 5% sample (to be refreshed on full data)
- Test macro-F1: LightGBM 0.763, logistic regression 0.717, majority class 0.214.
- Cost per incident on the test split under the min-cost rule: LightGBM 1.08, logistic regression 1.25, escalate everything 1.59. The plain argmax rule costs 2.50, which is worse than escalating everything.
- The six encoded alert-title and detector columns carry most of the model's gain. Ablation runs (scripts/ablate.py) show how much accuracy depends on them.

## Alternatives considered
- XGBoost: similar accuracy on tabular data, but LightGBM trains faster on millions of rows.
- Random forest: slower, usually weaker on this kind of data.
- Neural networks: no clear gain on tabular data, and harder to explain.
- Class weights: would distort the probabilities that the cost rule depends on.

## Consequences
- The cost matrix values are illustrative assumptions, not measured costs. Results are reported across a range of miss penalties.
- Probabilities feed the cost rule directly, so their calibration is checked in scripts/evaluate.py before the cost numbers are trusted.
- Macro-F1 and cost per incident can disagree. Both are reported.
- The model leans on alert title and detector, so new titles and detectors are a risk. In the full-data test window about 8% of incidents had a never-seen dominant alert title and they were not scored worse (see README), but they had a lower threat share, so this stays a monitored risk and not a measured one. Drift monitoring should track the unseen-value rate.
