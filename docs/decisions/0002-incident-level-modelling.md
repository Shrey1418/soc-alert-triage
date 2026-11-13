# ADR 0002: Incident-level modelling with a time-based split

Status: accepted

## Context
GUIDE rows are evidence rows. Evidence rolls up to alerts, and alerts roll up to incidents. Analysts triage incidents, and in the 5% sample every incident has exactly one grade (0 incidents with mixed labels). About 0.5% of rows have no label.

## Decision
- One training row per incident, keyed by (OrgId, IncidentId). IncidentId alone is not unique across organisations.
- Features are aggregates over the incident: counts of rows, alerts, detectors, alert titles, MITRE techniques, distinct entities, category shares, duration and start-time features.
- Incidents are split by start time: the oldest 70% train, the next 15% validate, the newest 15% test.
- Incidents with no label are dropped from training and evaluation.
- OrgId is not a model feature.

## Why
- A model that predicts per evidence row would score the same incident many times and would not match the analyst's task.
- A random split puts incidents from the same period in train and test, which inflates scores. A time split matches how the model is used: trained on the past, applied to new incidents.
- OrgId would let the model memorise organisations instead of learning threat patterns.

## Consequences
- Class mix and feature distributions may differ between periods. The split summary reports this per split.
- Organisation-level differences can still leak signal through other features. Group-aware validation is a possible later check.
- Full-data features need memory care, so columns are read selectively and low-cardinality text is stored as categoricals.