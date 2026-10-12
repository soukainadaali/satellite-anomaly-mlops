from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
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

data_asset = Data(
    name="smap_msl_raw",
    description="NASA SMAP/MSL raw telemetry data, version 1",
    path="azureml://datastores/satellite_data_store/paths/",
    type=AssetTypes.URI_FOLDER, #URI stands for Uniform Resource Identifier
)

ml_client.data.create_or_update(data_asset)