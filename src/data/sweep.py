from azure.ai.ml import command, Input, Output, MLClient
from azure.ai.ml.sweep import Choice
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""

CHANNEL_NAME = "E-13"          # P-1, E-13, F-7
DATA_ASSET = "smap_msl_processed:3"  
COMPUTE_TARGET = "anomaly-compute-cluster"
ENVIRONMENT = "satellite-anomaly-training-env@latest"


def main():
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        tenant_id=TENANT_ID,
    )

    base_job = command(
        code="../training",  
        command=(
            "python train.py "
            f"--channel_name {CHANNEL_NAME} "
            "--data_path ${{inputs.data_path}} "
            "--model_output ${{outputs.model_output}} "
            "--epochs 20 "
            "--batch_size ${{inputs.batch_size}} "
            "--lstm_units ${{inputs.lstm_units}}"
        ),
        environment=ENVIRONMENT,
        compute=COMPUTE_TARGET,
        inputs={
            "data_path": Input(type="uri_folder", path=f"azureml:{DATA_ASSET}"),
            "batch_size": 32,   
            "lstm_units": 64,
        },
        outputs={
            "model_output": Output(type="uri_folder"),
        },
        display_name=f"{CHANNEL_NAME}-sweep-trial",
    )

    sweep_job = base_job.sweep(
        compute=COMPUTE_TARGET,
        sampling_algorithm="grid",
        primary_metric="final_val_loss",  
        goal="Minimize",
    )
    sweep_job.search_space = {
        "batch_size": Choice([16, 32, 64]),
        "lstm_units": Choice([32, 64, 128]),
    }
    sweep_job.set_limits(
        max_total_trials=9,      # full grid: 3 x 3 combinations
        max_concurrent_trials=1,  # cluster's max node count is 1 — trials run sequentially
        timeout=10800,           # 3 hours safety cap
    )
    sweep_job.display_name = f"{CHANNEL_NAME}-sweep"
    sweep_job.experiment_name = "satellite-anomaly-detection"

    returned_job = ml_client.jobs.create_or_update(sweep_job)
    print(f"Sweep job submitted: {returned_job.name}")
    print(f"Studio URL: {returned_job.studio_url}")


if __name__ == "__main__":
    main()