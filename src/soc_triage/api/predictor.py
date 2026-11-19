import joblib
import pandas as pd

from soc_triage.data.schema import TARGET
from soc_triage.features.aggregate import build_incident_table
from soc_triage.models.cost import cost_matrix, min_cost_predictions
from soc_triage.models.train import transform_table
from soc_triage.utils.exceptions import DataValidationError

ACTIONS = {
    "TruePositive": "escalate",
    "BenignPositive": "review_later",
    "FalsePositive": "dismiss",
}


class Predictor:
    def __init__(self, model_path, cfg):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.encoder = bundle["encoder"]
        self.classes = list(bundle["classes"])
        self.feature_names = bundle["feature_names"]
        self.cost = cost_matrix(cfg["cost"], self.classes)
        self.entity_candidates = cfg["features"]["entity_columns"]
        self.category_candidates = cfg["features"]["category_columns"]

    def predict_incidents(self, evidence):
        df = pd.DataFrame(evidence)
        df[TARGET] = None
        entity_cols = [c for c in self.entity_candidates if c in df.columns]
        category_cols = [c for c in self.category_candidates if c in df.columns]

        table = build_incident_table(
            df, entity_cols, category_cols, top_n=1000, max_categories=1000
        )
        if table["start_time"].isna().any():
            raise DataValidationError("Timestamp could not be parsed for at least one incident")

        X = transform_table(table, self.encoder, self.feature_names)
        proba = self.model.predict_proba(X)
        decisions = min_cost_predictions(proba, self.cost, self.classes)

        results = []
        for i, decision in enumerate(decisions):
            decision = str(decision)
            rounded = [round(float(p), 4) for p in proba[i]]
            results.append(
                {
                    "OrgId": int(table.loc[i, "OrgId"]),
                    "IncidentId": int(table.loc[i, "IncidentId"]),
                    "decision": decision,
                    "action": ACTIONS[decision],
                    "probabilities": dict(zip(self.classes, rounded, strict=True)),
                }
            )
        return results
