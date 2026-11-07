import json, os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).resolve().parent.parent
MODELS = Path(os.getenv("MODELS_DIR", str(BASE / "models")))
REGISTRY: dict[str, dict[str, Any]] = {}
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))


def preprocess(image, cfg):
    size = cfg.get("input_size", 224)
    h, w = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))

    arr = np.asarray(
        ImageOps.exif_transpose(image)
        .convert("RGB")
        .resize((w, h), Image.Resampling.LANCZOS),
        dtype=np.float32
    )

    mode = cfg.get("preprocessing", "rescale_0_1")

    if mode in {"rescale_0_1", "rescale_255"}:
        arr /= 255.0

    elif mode == "rescale_minus1_1":
        arr = arr / 127.5 - 1.0

    elif mode in {"tf_efficientnet", "tf_mobilenet_v2", "tf_densenet"}:
        import tensorflow as tf

        fn = {
            "tf_efficientnet": tf.keras.applications.efficientnet.preprocess_input,
            "tf_mobilenet_v2": tf.keras.applications.mobilenet_v2.preprocess_input,
            "tf_densenet": tf.keras.applications.densenet.preprocess_input
        }[mode]

        arr = fn(arr)

    elif mode != "none":
        raise ValueError(f"Unknown preprocessing: {mode}")

    return np.expand_dims(arr, 0)


def load_models():
    REGISTRY.clear()
    MODELS.mkdir(parents=True, exist_ok=True)
    try:
        import tensorflow as tf
    except Exception as e:
        print("TensorFlow unavailable:", e); return
    for path in MODELS.iterdir():
        if path.suffix.lower() not in {".keras", ".h5"}: continue
        cfg_path = path.with_suffix(".json")
        if not cfg_path.exists():
            print(f"Skipping {path.name}: missing {cfg_path.name}"); continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            classes = cfg["classes"]
            if not isinstance(classes, list) or not classes: raise ValueError("'classes' must be a non-empty list")
            model = tf.keras.models.load_model(path, compile=False)
            out_shape = model.output_shape
            if isinstance(out_shape, list) or int(out_shape[-1]) != len(classes):
                raise ValueError(f"Output shape {out_shape} does not match {len(classes)} configured classes")
            REGISTRY[path.name] = {"model": model, "config": cfg, "classes": classes,
                                   "name": cfg.get("display_name", path.stem.replace("_", " "))}
            print("Loaded:", path.name)
        except Exception as e: print(f"Could not load {path.name}: {e}")

@asynccontextmanager
async def lifespan(app):
    load_models()
    yield

app = FastAPI(title="Oral Classification API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS","http://localhost:5173,http://127.0.0.1:5173").split(","),
                   allow_credentials=True, allow_methods=["GET","POST"], allow_headers=["*"])

@app.get("/")
def root(): return {"message":"Oral Classification API","docs":"/docs"}

@app.get("/api/health")
def health(): return {"status":"ok","loaded_models":len(REGISTRY)}

@app.get("/api/models")
def models():
    return {"models":[{"id":k,"name":v["name"]} for k,v in REGISTRY.items()]}

def prediction_scores(raw, mode):
    scores = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(scores)): raise ValueError("Model returned invalid scores")
    if mode == "probabilities": return np.clip(scores, 0, 1)
    if mode == "sigmoid": return 1/(1+np.exp(-np.clip(scores,-80,80)))
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    return exp / exp.sum()

def infer(record, image):
    cfg = record["config"]
    batch = preprocess(image, cfg)
    raw = record["model"].predict(batch, verbose=0)
    scores = prediction_scores(raw, cfg.get("output_mode","softmax"))
    classes = record["classes"]
    if len(scores) != len(classes): raise ValueError("Prediction output does not match configured classes")
    if cfg.get("task","single_label") == "multi_label":
        threshold = float(cfg.get("threshold",0.5))
        items = [{"label":classes[i],"confidence":float(scores[i])} for i in range(len(classes)) if scores[i] >= threshold]
        items.sort(key=lambda x:x["confidence"], reverse=True)
        best = items[0] if items else {"label":"No class above threshold","confidence":float(scores.max())}
        return {**best,"all_predictions":items,"task":"multi_label"}
    idx = int(np.argmax(scores))
    order = np.argsort(scores)[::-1][:min(5,len(scores))]
    return {"label":classes[idx],"confidence":float(scores[idx]),
            "all_predictions":[{"label":classes[i],"confidence":float(scores[i])} for i in order],
            "task":"single_label"}

@app.post("/api/classify")
async def classify(model_id: str = Form(...), files: list[UploadFile] = File(...)):
    if model_id not in REGISTRY: raise HTTPException(400, "Model is not loaded. Check the model file and matching JSON config, then restart FastAPI.")
    if not files: raise HTTPException(400, "Upload at least one image.")
    record = REGISTRY[model_id]
    output = []
    for f in files:
        if Path(f.filename or "").suffix.lower() not in {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}:
            raise HTTPException(415, f"Unsupported image type: {f.filename}")
        data = await f.read()
        if not data: raise HTTPException(400, f"Empty file: {f.filename}")
        if len(data) > MAX_MB*1024*1024: raise HTTPException(413, f"{f.filename} exceeds {MAX_MB} MB")
        try:
            im = Image.open(BytesIO(data)); im.verify(); im = Image.open(BytesIO(data))
        except (UnidentifiedImageError, OSError): raise HTTPException(400, f"Invalid image: {f.filename}")
        try: result = infer(record, im)
        except Exception as e: raise HTTPException(500, f"Inference failed for {f.filename}: {e}")
        output.append({"filename":f.filename, **result})
    return {"model_id":model_id,"model_name":record["name"],"predictions":output}
