from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


def register_dataset_via_sdk():
    SUBSCRIPTION_ID = ""
    RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
    WORKSPACE_NAME = "satellite-anomaly-ml"
    TENANT_ID = ""

    local_processed_path = Path(__file__).resolve().parent / "processed"

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        tenant_id=TENANT_ID,
    )

    my_data = Data(
        name="smap_msl_processed",
        version="3",
        description="Processed NASA SMAP/MSL telemetry features, uploaded via Azure ML SDK.",
        type=AssetTypes.URI_FOLDER,
        path=str(local_processed_path),  
    )

    registered_data = ml_client.data.create_or_update(my_data)
    print(f"Asset ID: {registered_data.id}")
    print(f"Asset path (where it landed in the workspace's storage): {registered_data.path}")


if __name__ == "__main__":
    register_dataset_via_sdk()