import json
import os
import time
from pathlib import Path

from azure.ai.ml import command, Input, Output, MLClient
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""

CHANNEL_NAME = "P-1"
DATA_ASSET = "smap_msl_processed:3"
LABELS_ASSET = "labeled_anomalies:1"
COMPUTE_TARGET = "anomaly-compute-cluster"
ENVIRONMENT = "satellite-anomaly-training-env@latest"

TRAINING_CODE = "../src/training"
EVALUATION_CODE = "../src/evaluation"

ENDPOINT_NAME = f"anomaly-{CHANNEL_NAME.lower()}-batch-endpoint"
MODEL_NAME = f"anomaly_model_{CHANNEL_NAME}"

RAW_WATCH_DIR = Path(__file__).resolve().parent.parent / "src" / "data" / "raw"
X_TEST_LOCAL_PATH = Path(__file__).resolve().parent.parent / "src" / "data" / "processed" / CHANNEL_NAME / "X_test.npy"

POLL_SECONDS = 15


def get_ml_client():
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        tenant_id=TENANT_ID,
    )


def submit_pipeline(ml_client):
    train_component = command(
        name="train_step", display_name="Train", code=TRAINING_CODE,
        command=(
            "python train.py "
            f"--channel_name {CHANNEL_NAME} "
            "--data_path ${{inputs.data_path}} "
            "--model_output ${{outputs.model_output}} "
            "--epochs 20 --batch_size 32 --lstm_units 64"
        ),
        environment=ENVIRONMENT,
        inputs={"data_path": Input(type="uri_folder")},
        outputs={"model_output": Output(type="uri_folder")},
    )
    evaluate_component = command(
        name="evaluate_step", display_name="Evaluate", code=EVALUATION_CODE,
        command=(
            "python evaluate_step.py "
            f"--channel_name {CHANNEL_NAME} "
            "--data_path ${{inputs.data_path}} "
            "--model_path ${{inputs.model_path}} "
            "--labeled_anomalies_path ${{inputs.labeled_anomalies_path}} "
            "--std_multiplier 3.0 "
            "--metrics_output ${{outputs.metrics_output}}"
        ),
        environment=ENVIRONMENT,
        inputs={
            "data_path": Input(type="uri_folder"),
            "model_path": Input(type="uri_folder"),
            "labeled_anomalies_path": Input(type="uri_file"),
        },
        outputs={"metrics_output": Output(type="uri_folder")},
    )

    @pipeline(compute=COMPUTE_TARGET, experiment_name="satellite-anomaly-detection",
              display_name=f"{CHANNEL_NAME}-auto-pipeline")
    def anomaly_pipeline(data_path, labeled_anomalies_path):
        train_job = train_component(data_path=data_path)
        evaluate_job = evaluate_component(
            data_path=data_path,
            model_path=train_job.outputs.model_output,
            labeled_anomalies_path=labeled_anomalies_path,
        )
        return {"model_output": train_job.outputs.model_output, "metrics_output": evaluate_job.outputs.metrics_output}

    pipeline_job = anomaly_pipeline(
        data_path=Input(type="uri_folder", path=f"azureml:{DATA_ASSET}"),
        labeled_anomalies_path=Input(type="uri_file", path=f"azureml:{LABELS_ASSET}"),
    )
    return ml_client.jobs.create_or_update(pipeline_job)


def wait_for_job(ml_client, job_name):
    print(f"  Waiting for job '{job_name}' to complete...")
    while True:
        job = ml_client.jobs.get(job_name)
        if job.status in ("Completed", "Failed", "Canceled"):
            print(f"  Job finished with status: {job.status}")
            return job.status
        time.sleep(POLL_SECONDS)


def get_current_production_f1(ml_client, model_name):
    try:
        versions = list(ml_client.models.list(name=model_name))
    except Exception:
        return None
    if not versions:
        return None
    latest = max(versions, key=lambda m: int(m.version))
    f1_tag = latest.tags.get("f1")
    return float(f1_tag) if f1_tag is not None else None


def register_if_better(ml_client, job_name):
    download_path = Path("downloaded_outputs") / job_name
    ml_client.jobs.download(name=job_name, download_path=str(download_path), output_name="metrics_output")
    metrics_file = download_path / "named-outputs" / "metrics_output" / "metrics.json"
    with open(metrics_file) as f:
        metrics = json.load(f)

    new_f1 = metrics["f1"]
    current_f1 = get_current_production_f1(ml_client, MODEL_NAME)

    if current_f1 is None or new_f1 > current_f1:
        job_model_path = f"azureml://jobs/{job_name}/outputs/model_output/paths/model_{CHANNEL_NAME}.keras"
        model_entity = Model(
            path=job_model_path, name=MODEL_NAME, type=AssetTypes.CUSTOM_MODEL,
            description=f"Auto-registered from job {job_name}.",
            tags={"channel": CHANNEL_NAME, "f1": str(new_f1), "precision": str(metrics["precision"]),
                  "recall": str(metrics["recall"]), "threshold": str(metrics["threshold"]), "source_job": job_name},
        )
        registered = ml_client.models.create_or_update(model_entity)
        print(f"  Registered {registered.name} v{registered.version} (f1={new_f1:.4f})")
        return True
    else:
        print(f"  New f1={new_f1:.4f} does not beat production f1={current_f1:.4f} -- skipping registration.")
        return False


def invoke_batch_endpoint(ml_client):
    job = ml_client.batch_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        input=Input(type=AssetTypes.URI_FILE, path=str(X_TEST_LOCAL_PATH)),
    )
    print(f"  Batch job invoked: {job.name}")
    return job.name


def run_full_chain(ml_client):
    print("New file detected -- starting full chain.")
    print("Step 1/3: submitting train+evaluate pipeline...")
    pipeline_job = submit_pipeline(ml_client)
    status = wait_for_job(ml_client, pipeline_job.name)
    if status != "Completed":
        print(f"Pipeline did not complete successfully (status={status}). Stopping chain.")
        return

    print("Step 2/3: checking whether to register the new model...")
    register_if_better(ml_client, pipeline_job.name)

    print("Step 3/3: invoking batch endpoint for fresh predictions...")
    batch_job_name = invoke_batch_endpoint(ml_client)
    batch_status = wait_for_job(ml_client, batch_job_name)

    if batch_status == "Completed":
        print("\nDone. predictions.csv is ready -- check it under:")
        print(f"  ml.azure.com -> Endpoints -> {ENDPOINT_NAME} -> Jobs -> {batch_job_name} -> Outputs + logs")
    else:
        print(f"\nBatch job did not complete successfully (status={batch_status}).")


def main():
    ml_client = get_ml_client()

    RAW_WATCH_DIR.mkdir(parents=True, exist_ok=True)
    known_files = set(os.listdir(RAW_WATCH_DIR))
    print(f"Watching {RAW_WATCH_DIR} for new files... (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(POLL_SECONDS)
            current_files = set(os.listdir(RAW_WATCH_DIR))
            new_files = current_files - known_files
            if new_files:
                print(f"New file(s) detected: {new_files}")
                run_full_chain(ml_client)
                known_files = current_files
                print(f"\nWatching {RAW_WATCH_DIR} for new files... (Ctrl+C to stop)")
    except KeyboardInterrupt:
        print("\nStopped watching.")


if __name__ == "__main__":
    main()