from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""

CHANNEL_NAME = "P-1"
MODEL_NAME = f"anomaly_model_{CHANNEL_NAME}"
MODEL_VERSION = "1"
ENDPOINT_NAME = f"anomaly-{CHANNEL_NAME.lower()}-endpoint"
DEPLOYMENT_NAME = "blue"  # blue/green naming
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

    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description=f"Real-time anomaly scoring for channel {CHANNEL_NAME}",
        auth_mode="key",
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint '{ENDPOINT_NAME}' ready.")

    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model,
        environment=ENVIRONMENT,
        code_configuration=CodeConfiguration(
            code=SCORING_CODE_DIR,
            scoring_script="score.py",
        ),
        environment_variables={"ANOMALY_THRESHOLD": threshold},
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Deployment '{DEPLOYMENT_NAME}' ready.")

    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Traffic routed: 100% -> {DEPLOYMENT_NAME}")


if __name__ == "__main__":
    main()