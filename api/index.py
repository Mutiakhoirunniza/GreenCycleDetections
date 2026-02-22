import os
import time
import json
import numpy as np
from PIL import Image
from io import BytesIO

# --- LIGHTWEIGHT TFLITE RUNTIME ---
import tflite_runtime.interpreter as tflite
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "model.tflite")

# Global variables for TFLite
interpreter = None
input_details = None
output_details = None

def load_tflite_model():
    global interpreter, input_details, output_details
    if interpreter is None:
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            return False
        
        try:
            print(f"DEBUG: Loading TFLite model from {MODEL_PATH}")
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("DEBUG: TFLite Model loaded successfully!")
            return True
        except Exception as e:
            print(f"ERROR: Could not load TFLite model: {e}")
            return False
    return True

class_labels = ["kaca", "kertas", "logam", "plastik"]

def prepare_image(img_pil, target_size=(224, 224)):
    img_pil = img_pil.resize(target_size)
    img_array = np.array(img_pil).astype("float32")
    # EfficientNet B0 Scale factor: [(x / 255.0)] or similar is often handled inside the TFLite model 
    # if it was converted from a Keras model with a Rescaling layer.
    # We'll follow the standard preprocess_input: ((x / 1.0)) since EfNet usually takes [0, 255]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        img_pil = Image.open(BytesIO(contents)).convert("RGB")
        
        if not load_tflite_model():
            return {"status": "error_model", "message": "Model file (.tflite) not found or failed to load."}

        img_array = prepare_image(img_pil)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        pred_probs = interpreter.get_tensor(output_details[0]['index'])[0]
        
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
    load_tflite_model()
