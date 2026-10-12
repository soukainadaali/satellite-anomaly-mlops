from azure.ai.ml import command, Input, Output, MLClient
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""

CHANNEL_NAME = "F-7"  
DATA_ASSET = "smap_msl_processed:3"
LABELS_ASSET = "labeled_anomalies:1"
COMPUTE_TARGET = "anomaly-compute-cluster"
ENVIRONMENT = "satellite-anomaly-training-env@latest"

TRAINING_CODE = "../src/training"      
EVALUATION_CODE = "../src/evaluation"  


def main():
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        tenant_id=TENANT_ID,
    )

    train_component = command(
        name="train_step",
        display_name="Train",
        code=TRAINING_CODE,
        command=(
            "python train.py "
            f"--channel_name {CHANNEL_NAME} "
            "--data_path ${{inputs.data_path}} "
            "--model_output ${{outputs.model_output}} "
            "--epochs 20 --batch_size 16 --lstm_units 64"  # batch_size=16 per SD-20 sweep results
        ),
        environment=ENVIRONMENT,
        inputs={"data_path": Input(type="uri_folder")},
        outputs={"model_output": Output(type="uri_folder")},
    )

    evaluate_component = command(
        name="evaluate_step",
        display_name="Evaluate",
        code=EVALUATION_CODE,
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

    @pipeline(
        compute=COMPUTE_TARGET,
        experiment_name="satellite-anomaly-detection",
        display_name=f"{CHANNEL_NAME}-pipeline",
    )
    def anomaly_pipeline(data_path, labeled_anomalies_path):
        train_job = train_component(data_path=data_path)
        evaluate_job = evaluate_component(
            data_path=data_path,
            model_path=train_job.outputs.model_output,
            labeled_anomalies_path=labeled_anomalies_path,
        )
        # Exposed as named pipeline-level outputs so local_register.py can download them afterward
        return {
            "model_output": train_job.outputs.model_output,
            "metrics_output": evaluate_job.outputs.metrics_output,
        }

    pipeline_job = anomaly_pipeline(
        data_path=Input(type="uri_folder", path=f"azureml:{DATA_ASSET}"),
        labeled_anomalies_path=Input(type="uri_file", path=f"azureml:{LABELS_ASSET}"),
    )

    returned_job = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Pipeline submitted: {returned_job.name}")
    print(f"Studio URL: {returned_job.studio_url}")
    print(f"Once complete, run: python local_register.py --job_name {returned_job.name}")


if __name__ == "__main__":
    main()