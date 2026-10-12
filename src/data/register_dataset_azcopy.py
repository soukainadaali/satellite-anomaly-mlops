import subprocess
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

def run_azcopy_upload(local_path: Path, storage_account: str, container: str, target_folder: str, sas_token: str):
    formatted_sas = sas_token if sas_token.startswith("?") else f"?{sas_token}"
        
    destination_uri = f"https://{storage_account}.blob.core.windows.net/{container}/{target_folder}{formatted_sas}"    
    source_path = str(local_path)
    
    cmd = [
        "azcopy", "copy",
        source_path,
        destination_uri,
        "--recursive=true",
        "--overwrite=true"
    ]

    subprocess.run(cmd, check=True)
    print("AzCopy transfer is completed.")

def register_dataset_with_azcopy():
    SUBSCRIPTION_ID = ""
    RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
    WORKSPACE_NAME = "satellite-anomaly-ml"
    TENANT_ID = ""  

    storage_account_name = "satanomalymlopsdata"
    container_name = "processed"
    datastore_name = "satellite_data_store"

    sas_token = ""  # paste a SAS token locally, never commit it
    
    local_processed_path = Path(__file__).resolve().parent / "processed"
    
    # Upload files via AzCopy to storage container subfolder /v1
    run_azcopy_upload(
        local_path=local_processed_path,
        storage_account=storage_account_name,
        container=container_name,
        target_folder="v1",
        sas_token=sas_token
    )
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME,
            tenant_id=TENANT_ID
        )
    
    #Point Azure ML Data Asset directly to the AzCopy-uploaded cloud path
    cloud_path = f"azureml://datastores/{datastore_name}/paths/{container_name}/v1"
    
    my_data = Data(
       name="smap_msl_processed",
        version="1",
        description="Processed NASA SMAP/MSL telemetry features uploaded via AzCopy v10.",
        type=AssetTypes.URI_FOLDER,
        path=cloud_path
    )

    registered_data = ml_client.data.create_or_update(my_data)
    print(f"Asset ID: {registered_data.id}")

if __name__ == "__main__":
    register_dataset_with_azcopy()
