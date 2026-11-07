# Oral Classification — React + FastAPI

## 1) Add your trained model files
Copy these four files into `backend/models/`:
- DenseNet121_oral_diseases.keras
- EfficientNetB0_oral_diseases.keras
- MobileNetV2_oral_diseases.keras
- CNN_oral_diseases.keras

The model files themselves are not bundled; they were not attached to this chat.

## 2) Add one JSON sidecar per model
Each sidecar has the same basename, for example `EfficientNetB0_oral_diseases.json`.
See `backend/models/README.md` for schema and preprocessing options.
Use the exact class order, input image size, preprocessing and output activation used in training. The API deliberately skips models without config files instead of guessing.

## 3) Run backend (Windows PowerShell)
```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API docs: http://127.0.0.1:8000/docs

## 4) Run frontend in a second terminal
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

## Features
Light/dark mode, single/multiple upload toggle, drag-and-drop, hide upload panel when images are selected, image gallery, thumbnails, navigation, remove selected, clear all, model selector, per-image prediction/confidence/top classes.

Educational/research prototype only; not for clinical diagnosis.
