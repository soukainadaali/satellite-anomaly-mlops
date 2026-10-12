from azure.ai.ml import MLClient
from azure.ai.ml.entities import AzureBlobDatastore
from azure.identity import InteractiveBrowserCredential

SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"

credential = InteractiveBrowserCredential(
    tenant_id="" 
)

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

datastore = AzureBlobDatastore(
    name="satellite_data_store",
    description="Dedicated storage account for satellite telemetry project data",
    account_name="satanomalymlopsdata",
    container_name="raw",
)

ml_client.datastores.create_or_update(datastore)