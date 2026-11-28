def retrain_decision(features, prediction, unseen, current_size, cfg):
    need = cfg["min_current_incidents"]
    if current_size < need:
        return {
            "retrain": False,
            "enough_data": False,
            "reasons": [f"only {current_size} incidents in the window, need at least {need}"],
        }

    ignore = set(cfg["retrain"].get("ignore_features", []))
    counted = features[~features["feature"].isin(ignore)]
    alerts = counted[counted["status"] == "alert"]

    reasons = []
    if len(alerts) >= cfg["retrain"]["min_alert_features"]:
        names = ", ".join(alerts["feature"].head(5))
        reasons.append(f"{len(alerts)} features in alert, for example: {names}")
    if prediction["status"] == "alert":
        reasons.append(f"prediction drift PSI {prediction['psi']:.3f}")
    for column, entry in unseen.items():
        if entry["status"] == "alert":
            reasons.append(f"{column}: {entry['rate']:.1%} of incidents have a never-seen value")
    return {"retrain": bool(reasons), "enough_data": True, "reasons": reasons}
