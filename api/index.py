import os
import time
import json
import numpy as np
from PIL import Image
from io import BytesIO

# --- LIGHTWEIGHT ONNX RUNTIME ---
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "model.onnx")

# Global variables for ONNX
session = None

def load_onnx_model():
    global session
    if session is None:
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            return False
        
        try:
            print(f"DEBUG: Loading ONNX model from {MODEL_PATH}")
            # Use CPU execution provider for Vercel
            session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
            print("DEBUG: ONNX Model loaded successfully!")
            return True
        except Exception as e:
            print(f"ERROR: Could not load ONNX model: {e}")
            return False
    return True

class_labels = ["kaca", "kertas", "logam", "plastik"]

def prepare_image(img_pil, target_size=(224, 224)):
    img_pil = img_pil.resize(target_size)
    img_array = np.array(img_pil).astype("float32")
    # EfficientNet preprocessing: scale usually handled in the model or via simple 1/255
    # If the model expects [0, 255], we just expand dims.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        img_pil = Image.open(BytesIO(contents)).convert("RGB")
        
        if not load_onnx_model():
            return {"status": "error_model", "message": "Model file (.onnx) not found."}

        img_array = prepare_image(img_pil)
        
        # ONNX inference
        input_name = session.get_inputs()[0].name
        pred_probs = session.run(None, {input_name: img_array})[0][0]
        
        idx = np.argmax(pred_probs)
        label = class_labels[idx]
        confidence = float(pred_probs[idx])
        probs_dict = {class_labels[i]: float(pred_probs[i]) for i in range(len(class_labels))}
        
        return {
            "label": label, 
            "confidence": confidence, 
            "probabilities": probs_dict, 
            "time": round(time.time() - start_time, 3), 
            "status": "ok"
        }
    except Exception as e:
        return {"status": "error_server", "message": str(e)}

# Static assets
app.mount("/Assets", StaticFiles(directory="Assets"), name="Assets")
@app.get("/style.css")
async def get_css(): return FileResponse("style.css")
@app.get("/script.js")
async def get_js(): return FileResponse("script.js")
@app.get("/")
async def read_index(): return FileResponse("index.html")
@app.get("/api/health")
async def health(): return {"status": "ok", "model_exists": os.path.exists(MODEL_PATH)}

@app.on_event("startup")
async def startup_event():
    load_onnx_model()
