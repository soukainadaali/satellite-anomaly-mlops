from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


SUBSCRIPTION_ID = ""
RESOURCE_GROUP = "rg-satellite-anomaly-mlops"
WORKSPACE_NAME = "satellite-anomaly-ml"
TENANT_ID = ""

MODELS_TO_BACKUP = [
    ("anomaly_model_P-1", "1"),
    ("anomaly_model_F-7", "1"),
]

BACKUP_DIR = Path("model_backups")


def main():
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        tenant_id=TENANT_ID,
    )

    BACKUP_DIR.mkdir(exist_ok=True)

    for model_name, version in MODELS_TO_BACKUP:
        print(f"Downloading {model_name} v{version} ...")
        target = BACKUP_DIR / f"{model_name}_v{version}"
        ml_client.models.download(name=model_name, version=version, download_path=str(target))
        print(f"  Saved to {target}")

    print("\nDone. All models backed up locally under model_backups/.")


if __name__ == "__main__":
    main()