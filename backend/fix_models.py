
import json
import zipfile
import shutil
import os
import tempfile

MODELS_DIR = "models"

MODEL_FILES = [
    "CNN_oral_diseases.keras",
    "DenseNet121_oral_diseases.keras",
    "EfficientNetB0_oral_diseases.keras",
    "MobileNetV2_oral_diseases.keras",
]


def remove_null_quantization_config(obj):
    """Remove only quantization_config keys whose value is None."""
    if isinstance(obj, dict):
        if obj.get("quantization_config", "NOT_FOUND") is None:
            del obj["quantization_config"]

        for value in obj.values():
            remove_null_quantization_config(value)

    elif isinstance(obj, list):
        for item in obj:
            remove_null_quantization_config(item)


for filename in MODEL_FILES:
    model_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(model_path):
        print(f"Not found: {filename}")
        continue

    backup_path = model_path + ".backup"

    # Preserve the original model
    if not os.path.exists(backup_path):
        shutil.copy2(model_path, backup_path)

    temp_path = model_path + ".tmp"

    try:
        with zipfile.ZipFile(model_path, "r") as source:
            with zipfile.ZipFile(temp_path, "w") as target:
                for item in source.infolist():
                    data = source.read(item.filename)

                    if item.filename == "config.json":
                        config = json.loads(data)
                        remove_null_quantization_config(config)
                        data = json.dumps(config).encode("utf-8")

                    target.writestr(item, data)

        # Replace only after successfully creating the new archive
        with zipfile.ZipFile(temp_path, "r") as check:
            if "config.json" not in check.namelist():
                raise ValueError("config.json is missing")

        os.replace(temp_path, model_path)

        print(f"Fixed successfully: {filename}")

    except Exception as error:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        print(f"Failed: {filename} - {error}")
