# Oral Classification — React + FastAPI

## 1. Download the Trained Models

Download the trained model files from the following Kaggle links:

- **CNN Model:**  
  https://www.kaggle.com/code/badrhishamzahran/oral-diseases-classification-by-cnn/output

- **Pretrained Models (DenseNet121, EfficientNetB0, MobileNetV2):**  
  https://www.kaggle.com/code/badrhishamzahran/oral-diseases-three-pretrained-models/output

Copy the following four `.keras` files into `backend/models/`:

- `CNN_oral_diseases.keras`
- `DenseNet121_oral_diseases.keras`
- `EfficientNetB0_oral_diseases.keras`
- `MobileNetV2_oral_diseases.keras`

**Note:** Model files are not included in this repository and must be downloaded separately.

## 2. Set Up the Backend Environment

Open a terminal in the project directory:

```powershell
cd backend

# Create virtual environment
py -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## 3. Fix Model Compatibility

Some `.keras` model files may contain serialization configuration keys that are incompatible with the installed Keras version.

Run the following script from the `backend` directory:

```powershell
python fix_models.py
```

This script removes unsupported `quantization_config` entries when their values are `null` and creates `.backup` copies of the original model files before modifying them.

**Important:** Run this script before testing the models. Keep the backup files until you have confirmed that all models work correctly.

## 4. Create the Model JSON Configuration Files

Each model requires a JSON configuration file with the same basename as its `.keras` file.

Create the following four files inside `backend/models/`.

### CNN_oral_diseases.json

```json
{
  "display_name": "Custom CNN",
  "classes": [
    "Calculus",
    "Data caries",
    "Gingivitis",
    "Mouth Ulcer",
    "hypodontia"
  ],
  "input_size": 224,
  "preprocessing": "rescale_255",
  "task": "single_label",
  "output_mode": "probabilities"
}
```

### DenseNet121_oral_diseases.json

```json
{
  "display_name": "DenseNet121",
  "classes": [
    "Calculus",
    "Data caries",
    "Gingivitis",
    "Mouth Ulcer",
    "hypodontia"
  ],
  "input_size": 224,
  "preprocessing": "none",
  "task": "single_label",
  "output_mode": "probabilities"
}
```

### EfficientNetB0_oral_diseases.json

```json
{
  "display_name": "EfficientNetB0 - Best Model",
  "classes": [
    "Calculus",
    "Data caries",
    "Gingivitis",
    "Mouth Ulcer",
    "hypodontia"
  ],
  "input_size": 224,
  "preprocessing": "none",
  "task": "single_label",
  "output_mode": "probabilities"
}
```

### MobileNetV2_oral_diseases.json

```json
{
  "display_name": "MobileNetV2",
  "classes": [
    "Calculus",
    "Data caries",
    "Gingivitis",
    "Mouth Ulcer",
    "hypodontia"
  ],
  "input_size": 224,
  "preprocessing": "none",
  "task": "single_label",
  "output_mode": "probabilities"
}
```

**Important:** The class order must match the exact class-index order used during model training.

## 5. Test the Downloaded Models

Before starting the application, verify that all four models load successfully.

From the `backend` directory, run:

```powershell
python test_models.py
```

The script checks:

- Whether each model loads successfully.
- Whether the input shape is `(None, 224, 224, 3)`.
- Whether the output shape is `(None, 5)`.

**Expected result:** All four models should display `SUCCESS`.

If a model fails, check its file name, download integrity, and compatibility before proceeding.

## 6. Start the Backend API

After all model tests pass, start FastAPI:

```powershell
uvicorn app.main:app --reload
```

Backend API:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

Available models endpoint:

http://127.0.0.1:8000/api/models

## 7. Install and Start the Frontend

Open a **second terminal** in the project directory:

```powershell
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Open the application:

http://localhost:5173

## Features

- Light and dark mode.
- Single-image and multiple-image upload modes.
- Drag-and-drop image upload.
- Image gallery with thumbnails.
- Image navigation.
- Remove selected image and clear all images.
- Model selection.
- Per-image predictions.
- Confidence scores and top predicted classes.

## Important Notice

This project is an educational and research prototype. It is not intended for clinical diagnosis or as a substitute for professional dental examination.