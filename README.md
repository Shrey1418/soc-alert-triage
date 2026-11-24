# SOC Alert Triage

Predicts the triage grade of security incidents (TruePositive, BenignPositive, FalsePositive) on the Microsoft GUIDE dataset, and turns the predicted probabilities into an escalate, review-later or dismiss decision using an explicit cost matrix. Served through a FastAPI service.

Status: the data pipeline, model, evaluation, API and CI are done. Docker, AWS deployment and drift monitoring are not built yet (see Roadmap).

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
The 33 tests use synthetic data and take a few seconds.

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

## Project layout
```
src/soc_triage/
  data/       schema, validation, CSV to Parquet, incident-level sampling
  features/   incident aggregation, time features, target encoder, chunked build
  models/     time split, baselines, cost matrix, metrics, report tables, training
  api/        FastAPI app, request schemas, predictor
  utils/      config, logger, exceptions
scripts/      convert, build_features, train, evaluate, ablate, call_api
tests/        unit and integration tests
docs/decisions/  ADR 0001 (model choice), ADR 0002 (incident-level modelling)
```

## Limitations
- The cost matrix is an illustrative assumption. Results are shown across a range of missed-threat penalties.
- The model is a filter, not a precise detector: 30.9% of escalated incidents are real threats.
- About 40% of the saving depends on alert title and detector. About 8.0% of test incidents have a dominant alert title never seen in training (0.9% for the detector); for those the encoding falls back to the overall class mix.
- The validation period is about 1.5 days and the test period about 4 days, and the class mix moves between periods (FalsePositive is 27.6% in train, 41.3% in valid, 32.4% in test). Results may not hold for other periods.
- Generalisation to unseen organisations was not tested. The same organisations appear in train and test.
- Probabilities are over-confident in some ranges (see Calibration).
- 3.2% of incidents have no label and were dropped.
- The API smoke script uses a small sample that is mostly training data, so it checks the plumbing, not accuracy.

## Roadmap
Docker image, AWS deployment with Terraform, and drift monitoring on the alert title and detector distributions. None of these are built yet.
