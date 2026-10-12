import json
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""

MODEL_NAME = "anomaly_model_P-1"
MODEL_VERSION = "1"
METRICS_JSON_PATH = "downloaded_outputs/goofy_ball_5zl0g6y9w3/named-outputs/metrics_output/metrics.json"  


def main():
    with open(METRICS_JSON_PATH) as f:
        metrics = json.load(f)

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        tenant_id=TENANT_ID,
    )

    model = ml_client.models.get(name=MODEL_NAME, version=MODEL_VERSION)
    model.tags["threshold"] = str(metrics["threshold"])
    updated = ml_client.models.create_or_update(model)
    print(f"Updated {updated.name} v{updated.version} — threshold tag: {updated.tags.get('threshold')}")


if __name__ == "__main__":
    main()