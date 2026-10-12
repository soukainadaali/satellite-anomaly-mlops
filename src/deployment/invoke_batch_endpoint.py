from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from pathlib import Path


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""

ENDPOINT_NAME = "anomaly-f-7-batch-endpoint"

X_TEST_LOCAL_PATH = str(Path(__file__).resolve().parent.parent / "data" / "processed" / "F-7" / "X_test.npy")


def main():
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        tenant_id=TENANT_ID,
    )

    job = ml_client.batch_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        input=Input(type=AssetTypes.URI_FILE, path=X_TEST_LOCAL_PATH),
    )
    print(f"Batch job invoked: {job.name}")
    print("Check its status in ml.azure.com under Endpoints -> anomaly-f-7-batch-endpoint -> Jobs")


if __name__ == "__main__":
    main()