from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
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

env = Environment(
    name="satellite-anomaly-training-env",
    description="Training environment for satellite telemetry anomaly detection",
    build=BuildContext(path="."),  
)

ml_client.environments.create_or_update(env)