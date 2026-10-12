from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
    CodeConfiguration,
)
from azure.ai.ml.constants import BatchDeploymentOutputAction
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""

CHANNEL_NAME = "P-1"
MODEL_NAME = f"anomaly_model_{CHANNEL_NAME}"
MODEL_VERSION = "1"  
ENDPOINT_NAME = f"anomaly-{CHANNEL_NAME.lower()}-batch-endpoint"  # existing endpoint
DEPLOYMENT_NAME = "green"
COMPUTE_TARGET = "anomaly-compute-cluster"
ENVIRONMENT = "satellite-anomaly-training-env@latest"
SCORING_CODE_DIR = "."  


def main():
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        tenant_id=TENANT_ID,
    )

    model = ml_client.models.get(name=MODEL_NAME, version=MODEL_VERSION)
    threshold = model.tags.get("threshold", "0.05")

    # Note: no endpoint creation here -- anomaly-p-1-batch-endpoint already exists,
    # we're just adding a second deployment onto it.
    deployment = ModelBatchDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model,
        environment=ENVIRONMENT,
        code_configuration=CodeConfiguration(
            code=SCORING_CODE_DIR,
            scoring_script="score_batch.py",
        ),
        compute=COMPUTE_TARGET,
        environment_variables={"ANOMALY_THRESHOLD": threshold},
        settings=ModelBatchDeploymentSettings(
            instance_count=1,
            max_concurrency_per_instance=1,
            mini_batch_size=1,
            output_action=BatchDeploymentOutputAction.APPEND_ROW,
            output_file_name="predictions.csv",
        ),
    )
    ml_client.batch_deployments.begin_create_or_update(deployment).result()
    print(f"Batch deployment '{DEPLOYMENT_NAME}' ready.")
    print("Note: 'blue' is still the default deployment -- 'green' exists but won't")
    print("receive traffic until you explicitly switch the default (see switch_default.py).")


if __name__ == "__main__":
    main()