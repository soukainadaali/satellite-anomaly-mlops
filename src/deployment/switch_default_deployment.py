from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""

ENDPOINT_NAME = "anomaly-p-1-batch-endpoint"
NEW_DEFAULT = "green"  


def main():
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        tenant_id=TENANT_ID,
    )

    endpoint = ml_client.batch_endpoints.get(ENDPOINT_NAME)
    old_default = endpoint.defaults.get("deployment_name") if endpoint.defaults else None

    endpoint.defaults = {"deployment_name": NEW_DEFAULT}
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

    print(f"Default deployment switched: '{old_default}' -> '{NEW_DEFAULT}'")


if __name__ == "__main__":
    main()