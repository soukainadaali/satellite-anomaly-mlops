from azure.ai.ml import MLClient
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


for env in ml_client.environments.list():
    print(env.name)
