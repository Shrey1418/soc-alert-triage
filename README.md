# SOC Alert Triage

Predicts the triage grade of security incidents (TruePositive, BenignPositive, FalsePositive) on the Microsoft GUIDE dataset, and turns the predicted probabilities into an escalate, review-later or dismiss decision using an explicit cost matrix. Served through a FastAPI service.

Status: the data pipeline, model, evaluation, API, Docker image, local drift monitoring and CI are done. AWS deployment is not built yet (see Roadmap).

## Problem
Security analysts receive more incidents than they can review, and most turn out to be benign or false alarms. The goal is to put the likely real threats in front of analysts without letting real threats slip through. Missing a real threat is much more expensive than escalating a harmless incident, so the decision rule is cost-aware instead of "pick the most likely class".

## Data
Microsoft GUIDE security incident dataset (Kaggle, "Microsoft Security Incident Prediction"). The training file has 9,516,837 evidence rows. They roll up to alerts and then to 730,778 incidents, and each incident has one grade. About 3.2% of incidents have no label and are dropped. The data is not stored in this repository.

Incidents are split by start time, oldest to newest, so the test set is the newest period:

| Split | Incidents | Period | BenignPositive | FalsePositive | TruePositive |
|---|---|---|---|---|---|
| train | 494,976 | 2023-11-20 to 2024-06-12 | 50.7% | 27.6% | 21.7% |
| valid | 106,066 | 2024-06-12 to 2024-06-13 | 42.5% | 41.3% | 16.2% |
| test | 106,066 | 2024-06-13 to 2024-06-17 | 47.7% | 32.4% | 19.9% |

## Approach
- One row per incident, keyed by (OrgId, IncidentId). Features are aggregates over the incident: counts of evidence rows, alerts, detectors and MITRE techniques, distinct entities, category shares, duration and start-time features.
- The dominant alert title and dominant detector of each incident are target-encoded out of fold, fitted on the training split only, so an incident never sees its own label.
- OrgId and raw timestamps are not model features.
- LightGBM multiclass with early stopping on the validation split. Baselines: majority class and logistic regression. No class weights, so the probabilities stay usable for the cost rule.
- The decision picks the action with the lowest expected cost under a cost matrix. The plain most-likely-class rule is reported next to it.
- The API takes raw evidence rows and builds the features with the same code as training. A test checks that the API and the training pipeline give the same probabilities for the same incident.
- The full 9.5M-row feature build runs in 16 chunks grouped by incident, and a test checks that it matches the in-memory build.

## Results
Test split (newest period, 106,066 incidents):

| Model | Macro-F1 (most likely class) | Cost per incident (min-cost rule) |
|---|---|---|
| Escalate everything | - | 1.603 |
| Majority class | 0.215 | 1.603 |
| Logistic regression | 0.754 | 1.165 |
| LightGBM | 0.788 | 1.012 |

LightGBM is 36.9% cheaper than escalating everything, and 13% cheaper than logistic regression. The plain most-likely-class rule costs 2.147 per incident, which is worse than escalating everything, so the cost rule matters more than the model. Under the min-cost rule macro-F1 is 0.510, because the rule deliberately over-escalates.

What the min-cost rule does with the test incidents:

| Action | Incidents | Share | Real threats inside |
|---|---|---|---|
| escalate | 66,582 | 62.8% | 20,541 |
| review later | 29,668 | 28.0% | 487 |
| dismiss | 9,816 | 9.3% | 27 |

Of the 21,055 real threats, 97.6% are escalated, 2.3% go to review later and 0.13% are dismissed. Only 30.9% of escalated incidents are real threats, so this works as a filter that safely de-prioritises 37% of incidents. It is not a precise threat detector.

Cost matrix (illustrative assumptions, not measured costs; rows are the true grade):

| True grade | escalate | review later | dismiss |
|---|---|---|---|
| TruePositive | 0 | 20 | 50 |
| BenignPositive | 2 | 0 | 1 |
| FalsePositive | 2 | 1 | 0 |

Sensitivity to the cost of a missed threat (multiplier on the TruePositive row):

| Missed threat costs | Saving vs escalate everything | Incidents escalated | Real threats dismissed |
|---|---|---|---|
| 0.25x | 63.9% | 27.8% | 2.3% |
| 0.5x | 48.0% | 47.1% | 1.1% |
| 1x (config) | 36.9% | 62.8% | 0.13% |
| 2x | 32.9% | 69.2% | 0.04% |
| 4x | 29.7% | 72.7% | 0.005% |

Where the signal comes from (feature groups removed, model retrained, test split):

| Features | Macro-F1 | Cost | Saving vs escalate everything |
|---|---|---|---|
| All features | 0.788 | 1.012 | 36.9% |
| Without time-of-day features | 0.791 | 1.026 | 36.0% |
| Without alert title and detector | 0.700 | 1.224 | 23.6% |

The model still beats escalating everything without alert title and detector, but about 40% of the saving depends on them.

## Calibration
The TruePositive probability is well calibrated at the top (0.993 predicted vs 0.992 observed in the highest bin, 8,352 incidents) and over-confident in the middle (0.640 predicted vs 0.488 observed for incidents scored between 0.6 and 0.7). Between 0.04 and 0.10, where the escalate-or-clear boundary of the cost rule roughly sits (derived from the cost matrix, not measured), it over-estimates risk, for example 0.066 predicted vs 0.034 observed. The reported costs are measured on true labels, so they are valid. A recalibrated model could pick a better boundary.

## Quickstart (no dataset needed)
```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,api]"
python -m pytest -q
```
The tests use synthetic data and take a few seconds.

## Full pipeline (needs the GUIDE data)
Download the GUIDE training file into data/raw, then:
```
python scripts/convert_to_parquet.py train
python scripts/build_features.py --input data/processed/train.parquet --out data/processed/incidents_full.parquet --buckets 16
python scripts/train.py --input data/processed/incidents_full.parquet
python scripts/evaluate.py --input data/processed/incidents_full.parquet
python scripts/ablate.py --input data/processed/incidents_full.parquet
uvicorn soc_triage.api.main:app --port 8000
python scripts/call_api.py
```
Open http://127.0.0.1:8000/docs to try POST /predict. A request is a list of evidence rows; the response has one decision per incident with the three class probabilities.

## Docker
The image serves the saved model with the same code as the local API. It copies artifacts/model.joblib, which is not in the repository, so run the training step first. The serving requirements in requirements-api.txt are pinned to the versions of the training environment, so the saved model loads with the same library versions.
```
docker build -t soc-triage:0.1 .
docker run --rm -d --name soc-api -p 8000:8000 soc-triage:0.1
python scripts/call_api.py
docker stop soc-api
```
The image has a health check that calls /health every 30 seconds. The image is about 948 MB on disk (222 MB of compressed content).

## Project layout
```
src/soc_triage/
  data/       schema, validation, CSV to Parquet, incident-level sampling
  features/   incident aggregation, time features, target encoder, chunked build
  models/     time split, baselines, cost matrix, metrics, report tables, training
  api/        FastAPI app, request schemas, predictor
  monitoring/ drift metrics, retrain rules, shift simulation
  utils/      config, logger, exceptions
scripts/      convert, build_features, train, evaluate, ablate, call_api, run_drift_check, simulate_drift
tests/        unit and integration tests
docs/decisions/  ADR 0001 (model choice), ADR 0002 (incident-level modelling)
```

## Drift monitoring
A drift check compares a recent window with a reference window. For every model feature it computes the population stability index (PSI), and for numeric features the Kolmogorov-Smirnov distance. It also reports the share of incidents whose dominant alert title or detector never appeared in the reference, and the drift of the model's TruePositive probability. Thresholds are in configs/drift.yaml and are assumptions, not measured limits. A retrain decision is raised when prediction drift or an unseen-value rate reaches its alert level, or when at least 5 features alert. It is a function that returns yes or no with the reasons. No scheduled job and no automatic retraining exist yet.

Calendar features (start hour, day of week, weekend flag) are shown but not counted for retraining. Short windows cover different weekdays, so their PSI measures calendar coverage and not a change in the data. In the check below the reference window (Wednesday to Thursday) has no weekend and the current window (Thursday to Monday) has one.
```
python scripts/run_drift_check.py
python scripts/run_drift_check.py --reference train
python scripts/simulate_drift.py
```

Real windows, with the validation window as reference and the test window as current:

| Check | Result |
|---|---|
| TruePositive probability (prediction drift) | PSI 0.199 (warn); incidents escalated 59.4% in the reference, 62.8% now |
| Dominant alert title never seen in the reference | 8.8% (ok) |
| Dominant detector never seen in the reference | 1.4% (ok) |
| PSI of dominant alert title and of detector | 0.141 and 0.110 (both warn) |
| Retrain decision | no |

The class mix also moved between the two windows (TruePositive 16.2% to 19.9%), which explains part of the prediction drift.

Simulated shift on the test window: 30% of incidents get an alert title never seen in training, and the number of evidence rows is multiplied by 2.5. The labels are unchanged, so this tests whether the alarm fires and how much the model suffers, not how real drift behaves.

| Window | Unseen alert titles | Prediction PSI | Incidents escalated | Cost per incident | Saving vs escalate everything | Retrain |
|---|---|---|---|---|---|---|
| Test, as is | 8.8% | 0.199 | 62.8% | 1.012 | 36.9% | no |
| Test, simulated shift | 36.1% | 0.588 | 69.1% | 1.081 | 32.6% | yes |

The shift raises the cost per incident by 6.9%, and the model still beats escalating everything (1.603). Cost needs the true grade, which a live system would typically get only after analysts close an incident, so input drift and prediction drift are the signals that can fire first.

## Documentation and related projects
- Design decisions: [ADR 0001, model choice and cost-aware decisions](docs/decisions/0001-model-choice-lightgbm.md) and [ADR 0002, incident-level modelling and time-based split](docs/decisions/0002-incident-level-modelling.md)
- Exploration: [EDA notebook](notebooks/01_eda.ipynb)
- Other projects, built with the same modular structure:
  - [Loan Default Risk Prediction](https://github.com/Shrey1418/loan-default-risk-prediction): credit card default model with SHAP explanations and a profit-based approval threshold
  - [Hospital Readmission Risk Prediction](https://github.com/Shrey1418/hospital-readmission-risk-prediction): 30-day readmission model with calibrated probabilities and a cost-based call policy

## Limitations
- The cost matrix is an illustrative assumption. Results are shown across a range of missed-threat penalties.
- The model is a filter, not a precise detector: 30.9% of escalated incidents are real threats.
- About 40% of the saving depends on alert title and detector. About 8.0% of test incidents have a dominant alert title never seen in training (0.9% for the detector), and the encoding falls back to the overall class mix for them. In this test window those incidents scored better than the rest (macro-F1 0.879 vs 0.782 with the most-likely-class rule, cost 0.277 vs 1.076 with the min-cost rule), but only 9.4% of them are real threats against 20.8% for the rest, so the two groups are not comparable. This is one window, not proof that new alert titles are safe.
- The validation period is about 1.5 days and the test period about 4 days, and the class mix moves between periods (FalsePositive is 27.6% in train, 41.3% in valid, 32.4% in test). Results may not hold for other periods.
- Generalisation to unseen organisations was not tested. The same organisations appear in train and test.
- Probabilities are over-confident in some ranges (see Calibration).
- 3.2% of incidents have no label and were dropped.
- The API smoke script uses a small sample that is mostly training data, so it checks the plumbing, not accuracy.

## Roadmap
AWS deployment with Terraform, and running the drift check on a schedule with alarms. Neither is built yet.
