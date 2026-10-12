import argparse
import json
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""


def parse_args():
    parser = argparse.ArgumentParser(description="Register the trained model only if its f1 beats the current production model.")
    parser.add_argument("--channel_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Folder containing model_<channel>.keras (output of the train step)")
    parser.add_argument("--metrics_input", type=str, required=True, help="Folder containing metrics.json (output of the evaluate step)")
    return parser.parse_args()


def get_current_production_f1(ml_client: MLClient, model_name: str):
    
    try:
        versions = list(ml_client.models.list(name=model_name))
    except Exception:
        return None

    if not versions:
        return None

    latest = max(versions, key=lambda m: int(m.version))
    f1_tag = latest.tags.get("f1")
    return float(f1_tag) if f1_tag is not None else None


def main():
    args = parse_args()

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        tenant_id=TENANT_ID,
    )

    with open(Path(args.metrics_input) / "metrics.json") as f:
        metrics = json.load(f)
    new_f1 = metrics["f1"]

    model_name = f"anomaly_model_{args.channel_name}"
    current_f1 = get_current_production_f1(ml_client, model_name)

    if current_f1 is None:
        print(f"[{args.channel_name}] no existing production model found — registering as first version.")
        should_register = True
    elif new_f1 > current_f1:
        print(f"[{args.channel_name}] new f1={new_f1:.4f} beats current production f1={current_f1:.4f} — registering.")
        should_register = True
    else:
        print(f"[{args.channel_name}] new f1={new_f1:.4f} does NOT beat current production f1={current_f1:.4f} — skipping registration.")
        should_register = False

    if should_register:
        model_file = Path(args.model_path) / f"model_{args.channel_name}.keras"
        model_entity = Model(
            path=str(model_file),
            name=model_name,
            type=AssetTypes.CUSTOM_MODEL,
            description=f"LSTM anomaly-detection model for channel {args.channel_name}",
            tags={
                "channel": args.channel_name,
                "f1": str(new_f1),
                "precision": str(metrics["precision"]),
                "recall": str(metrics["recall"]),
            },
        )
        registered = ml_client.models.create_or_update(model_entity)
        print(f"[{args.channel_name}] registered model: {registered.name}, version {registered.version}")


if __name__ == "__main__":
    main()