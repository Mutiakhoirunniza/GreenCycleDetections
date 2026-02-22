
import os
import time
import json
import h5py
import numpy as np
from PIL import Image
from io import BytesIO

# --- CRITICAL: Set environment variables BEFORE importing tensorflow ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- ROBUST COMPATIBILITY PATCHER ---

class ShapeString(str):
    """A string that mimics a TensorShape to avoid Keras 2 crashes."""
    def as_list(self): return [None, 224, 224, 3]

def deep_clean_config(config, depth=0):
    if depth > 50: # Safety break
        return config
        
    if not isinstance(config, (dict, list)):
        return config
    
    if isinstance(config, list):
        return [deep_clean_config(v, depth + 1) for v in config]
    
    # It's a dict, create a copy to avoid mutation issues
    config = dict(config)
    
    # 1. Patch DTypePolicy (K3) -> string (K2)
    if config.get("class_name") == "DTypePolicy":
        return config.get("config", {}).get("name", "float32")
    
    # 2. Patch Layer Configs
    cls_name = config.get("class_name")
    if "config" in config and isinstance(config["config"], dict):
        cfg = dict(config["config"])
        
        # Patch InputLayer / Shapes
        if cls_name == "InputLayer":
            # Keras 2 expects 'batch_input_shape'
            if "batch_shape" in cfg:
                cfg["batch_input_shape"] = cfg.pop("batch_shape")
        
        # Patch any other layers that might have batch_shape in their config
        if "batch_shape" in cfg and not cls_name == "InputLayer":
             cfg["batch_input_shape"] = cfg.pop("batch_shape")
        
        # Patch axis in Normalization/BN
        if cls_name in ["BatchNormalization", "Normalization"]:
            if "axis" in cfg and isinstance(cfg["axis"], (list, tuple)):
                cfg["axis"] = cfg["axis"][0] if len(cfg["axis"]) == 1 else tuple(cfg["axis"])

        # Patch dtypes and remove K3 junk
        for k in ["dtype", "compute_dtype", "variable_dtype"]:
            if k in cfg and isinstance(cfg[k], dict):
                cfg[k] = cfg[k].get("config", {}).get("name", "float32")
        
        for junk in ["sparse", "ragged", "module", "registered_name", "synchronized"]:
            cfg.pop(junk, None)
            config.pop(junk, None)
            
        config["config"] = cfg

    # 3. Patch Functional connectivity (inbound_nodes)
    if "inbound_nodes" in config and isinstance(config["inbound_nodes"], list):
        new_nodes = []
        for node in config["inbound_nodes"]:
            if isinstance(node, dict) and "args" in node:
                k2_node = []
                args = node.get("args", [])
                kwargs = node.get("kwargs", {})
                
                # Flatten single-element lists if they contain the real arguments
                if len(args) == 1 and isinstance(args[0], list) and not (len(args[0]) >= 3 and isinstance(args[0][0], str)):
                    args = args[0]

                for arg in args:
                    if isinstance(arg, dict) and arg.get("class_name") == "__keras_tensor__":
                        h = arg.get("config", {}).get("keras_history")
                        if h:
                            h = list(h)
                            while len(h) < 4: h.append({})
                            h[3] = kwargs
                            k2_node.append(h)
                    elif isinstance(arg, list) and len(arg) >= 3 and isinstance(arg[0], str):
                        h = list(arg)
                        while len(h) < 4: h.append({})
                        h[3] = kwargs
                        k2_node.append(h)
                if k2_node:
                    new_nodes.append(k2_node)
            else:
                new_nodes.append(node)
        config["inbound_nodes"] = new_nodes

    # 4. Recursion
    for k, v in config.items():
        config[k] = deep_clean_config(v, depth + 1)
            
    return config

# --- APP SETUP ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "model_effnetb0_best_tuned_final.h5")
model = None

def load_prediction_model():
    global model
    if model is None:
        try:
            print(f"DEBUG: Loading model with deep patches from {MODEL_PATH}")
            with h5py.File(MODEL_PATH, 'r') as f:
                cfg_str = f.attrs.get('model_config')
                if isinstance(cfg_str, bytes): cfg_str = cfg_str.decode('utf-8')
                full_config = json.loads(cfg_str)
            
            cleaned_config = deep_clean_config(full_config)
            
            # Use Fallbacks for K3 specific layers
            from tensorflow.keras.layers import Layer, Normalization, Rescaling
            custom_map = {
                "DTypePolicy": lambda **kwargs: "float32",
                "Normalization": Normalization,
                "Rescaling": Rescaling,
            }
            
            model = keras.models.model_from_config(cleaned_config, custom_objects=custom_map)
            model.load_weights(MODEL_PATH)
            print("DEBUG: Model loaded and weights assigned successfully!")
        except Exception as e:
            print(f"ERROR during model loading: {str(e)}")
            # Last ditch effort: load_model with custom_objects
            try:
                model = keras.models.load_model(MODEL_PATH, compile=False, custom_objects={"ShapeString": ShapeString})
            except:
                raise e
    return model

class_labels = ["kaca", "kertas", "logam", "plastik"]

def prepare_image(img_pil, target_size=(224, 224)):
    img_pil = img_pil.resize(target_size)
    img = img_pil.convert("RGB")
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def contains_face(img_pil):
    try:
        import cv2
        img_cv = np.array(img_pil.convert("RGB"))
        img_cv = cv2.resize(img_cv, (300, 300))
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50))
        return len(faces) > 0
    except Exception: return False

@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        img_pil = Image.open(BytesIO(contents)).convert("RGB")
        
        if contains_face(img_pil):
            return {"label": "manusia", "confidence": 1.0, "probabilities": {}, "time": round(time.time() - start_time, 3), "status": "error_human"}
        
        img_array = prepare_image(img_pil)
        predictor = load_prediction_model()
        pred_probs = predictor.predict(img_array)[0]
        
        idx = np.argmax(pred_probs)
        label = class_labels[idx]
        confidence = float(pred_probs[idx])
        probs_dict = {class_labels[i]: float(pred_probs[i]) for i in range(len(class_labels))}
        
        status = "ok"
        if confidence < 0.6: status = "uncertain"
        
        return {"label": label, "confidence": confidence, "probabilities": probs_dict, "time": round(time.time() - start_time, 3), "status": status}
    except Exception as e:
        print(f"PREDICTION ERROR: {str(e)}")
        return {"status": "error_server", "message": str(e), "label": "error", "confidence": 0, "time": 0, "probabilities": {}}

# Static assets
app.mount("/Assets", StaticFiles(directory="Assets"), name="Assets")
@app.get("/style.css")
async def get_css(): return FileResponse("style.css")
@app.get("/script.js")
async def get_js(): return FileResponse("script.js")
@app.get("/")
async def read_index(): return FileResponse("index.html")
@app.get("/api/health")
async def health(): return {"status": "ok", "model_path": MODEL_PATH, "exists": os.path.exists(MODEL_PATH)}

# --- PRE-LOAD MODEL ON STARTUP ---
# This makes the first prediction much faster
@app.on_event("startup")
async def startup_event():
    print("Pre-loading model for faster performance...")
    try:
        load_prediction_model()
        # Warm up the model with a dummy prediction
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        model.predict(dummy_input)
        print("Model pre-loaded and warmed up!")
    except Exception as e:
        print(f"Startup model load failed: {e}")
