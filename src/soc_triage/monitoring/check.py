from soc_triage.models.cost import cost_matrix
from soc_triage.models.train import ENCODE_COLS, numeric_feature_columns, transform_table
from soc_triage.monitoring.drift import feature_drift, prediction_drift, unseen_report
from soc_triage.monitoring.retrain import retrain_decision


def probabilities(bundle, table):
    X = transform_table(table, bundle["encoder"], bundle["feature_names"])
    return bundle["model"].predict_proba(X)


def run_drift_check(bundle, reference, pred_reference, current, cfg):
    drift_cfg = cfg["drift"]
    classes = list(bundle["classes"])
    cost = cost_matrix(cfg["cost"], classes)
    ignored = list(drift_cfg["retrain"].get("ignore_features", []))

    features = feature_drift(
        reference, current, numeric_feature_columns(reference), list(ENCODE_COLS), drift_cfg
    )
    features["counts_for_retrain"] = ~features["feature"].isin(ignored)
    unseen = unseen_report(reference, current, list(ENCODE_COLS), drift_cfg["unseen_rate"])
    prediction = prediction_drift(
        probabilities(bundle, pred_reference),
        probabilities(bundle, current),
        classes,
        cost,
        drift_cfg,
    )
    decision = retrain_decision(features, prediction, unseen, len(current), drift_cfg)
    return {
        "features": features,
        "unseen": unseen,
        "prediction": prediction,
        "decision": decision,
        "ignored": ignored,
    }


def format_report(result, top=12):
    features = result["features"]
    counted = features[features["counts_for_retrain"]]
    counts = counted["status"].value_counts().to_dict()
    flagged = features[features["status"] != "ok"].head(top)

    lines = [f"Feature status counts (without calendar features): {counts}"]
    if len(flagged):
        lines.append(flagged.to_string(index=False))
    if result["ignored"]:
        lines.append(
            "Shown but not counted for retraining (short windows cover different weekdays): "
            + ", ".join(result["ignored"])
        )
    lines.append("Values never seen in the reference:")
    for column, entry in result["unseen"].items():
        lines.append(f"  {column}: {entry['rate']:.1%} ({entry['status']})")

    p = result["prediction"]
    shares = p["escalated_share"]
    lines.append(
        f"Prediction drift: PSI {p['psi']:.3f} ({p['status']}), escalated share "
        f"{shares['reference']:.1%} -> {shares['current']:.1%}"
    )
    decision = result["decision"]
    lines.append(f"Retrain: {decision['retrain']}")
    lines.extend(f"  - {reason}" for reason in decision["reasons"])
    return "\n".join(lines)
