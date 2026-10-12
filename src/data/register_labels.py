from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""


def register_labeled_anomalies():
    # Assumes this script is run from src/data/, with raw/labeled_anomalies.csv alongside it
    local_csv_path = Path(__file__).resolve().parent / "raw" / "labeled_anomalies.csv"

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        tenant_id=TENANT_ID,
    )

    my_data = Data(
        name="labeled_anomalies",
        version="1",
        description="NASA SMAP/MSL ground-truth anomaly labels (chan_id, anomaly_sequences, class, num_values).",
        type=AssetTypes.URI_FILE,
        path=str(local_csv_path),
    )

    registered_data = ml_client.data.create_or_update(my_data)
    print(f"Asset ID: {registered_data.id}")
    print(f"Asset path: {registered_data.path}")


if __name__ == "__main__":
    register_labeled_anomalies()