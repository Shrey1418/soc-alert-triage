import argparse
import json

import httpx
import pandas as pd

from soc_triage.data.schema import INCIDENT_KEY, TARGET
from soc_triage.utils.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Send real sample incidents to the API")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--incidents", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_parquet(PROJECT_ROOT / "data/sample/train_tiny.parquet")
    keys = df[INCIDENT_KEY].drop_duplicates().head(args.incidents)
    chosen = df.merge(keys, on=INCIDENT_KEY)
    truth = chosen.groupby(INCIDENT_KEY)[TARGET].first()
    evidence = json.loads(chosen.drop(columns=[TARGET]).to_json(orient="records"))

    response = httpx.post(f"{args.url}/predict", json={"evidence": evidence}, timeout=60)
    response.raise_for_status()
    for item in response.json()["results"]:
        key = (item["OrgId"], item["IncidentId"])
        print(
            f"org={key[0]} incident={key[1]} -> {item['action']:<12}",
            item["probabilities"],
            "| true grade:",
            truth.get(key),
        )


if __name__ == "__main__":
    main()
