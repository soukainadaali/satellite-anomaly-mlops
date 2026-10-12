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

DOWNLOAD_DIR = "downloaded_outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Register a completed pipeline job's model if it's better than production, preserving native job lineage.")
    parser.add_argument("--job_name", type=str, required=True, help="The pipeline job name printed by pipeline.py, e.g. 'gentle_river_abc123'")
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

    # Only download metrics.json -- small, and we need its contents locally to decide
    # whether to register at all. The model file itself is NOT downloaded; we instead
    # register it directly from the job's own output path (see below), which preserves
    # Azure ML's native job -> model lineage link instead of breaking it via a
    # download-then-reupload round trip.
    download_path = Path(DOWNLOAD_DIR) / args.job_name
    print(f"Downloading metrics from job '{args.job_name}' to {download_path} ...")
    ml_client.jobs.download(name=args.job_name, download_path=str(download_path), output_name="metrics_output")

    metrics_dir = download_path / "named-outputs" / "metrics_output"
    metrics_file = metrics_dir / "metrics.json"
    with open(metrics_file) as f:
        metrics = json.load(f)

    channel_name = metrics["channel_name"]
    new_f1 = metrics["f1"]

    model_name = f"anomaly_model_{channel_name}"
    current_f1 = get_current_production_f1(ml_client, model_name)

    if current_f1 is None:
        print(f"[{channel_name}] no existing production model found — registering as first version.")
        should_register = True
    elif new_f1 > current_f1:
        print(f"[{channel_name}] new f1={new_f1:.4f} beats current production f1={current_f1:.4f} — registering.")
        should_register = True
    else:
        print(f"[{channel_name}] new f1={new_f1:.4f} does NOT beat current production f1={current_f1:.4f} — skipping registration.")
        should_register = False

    if should_register:
        # Reference the model directly at its location within the job's own output --
        # this is what lets Azure ML automatically show this model's lineage back to
        # the job (and the job's own input data asset), in Studio's Lineage graph.
        job_model_path = f"azureml://jobs/{args.job_name}/outputs/model_output/paths/model_{channel_name}.keras"

        model_entity = Model(
            path=job_model_path,
            name=model_name,
            type=AssetTypes.CUSTOM_MODEL,
            description=f"LSTM anomaly-detection model for channel {channel_name} (registered from pipeline job {args.job_name}, native lineage preserved).",
            tags={
                "channel": channel_name,
                "f1": str(new_f1),
                "precision": str(metrics["precision"]),
                "recall": str(metrics["recall"]),
                "threshold": str(metrics["threshold"]),
                "source_job": args.job_name,
            },
        )
        registered = ml_client.models.create_or_update(model_entity)
        print(f"Registered model: {registered.name}, version {registered.version}")
        print(f"Registered directly from job output — check Studio's Lineage tab to confirm the job link.")


if __name__ == "__main__":
    main()