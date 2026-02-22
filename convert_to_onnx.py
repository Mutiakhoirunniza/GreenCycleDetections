import os
import subprocess
import sys

def convert():
    h5_path = "Model/model_effnetb0_best_tuned_final.h5"
    onnx_path = "Model/model.onnx"

    print("--- GreenCycle Model Converter (ONNX Edition) ---")
    
    # 1. Install tf2onnx if not present
    try:
        import tf2onnx
        import tensorflow as tf
    except ImportError:
        print("Installing required tools (tensorflow-cpu, tf2onnx)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-cpu==2.15.0", "tf2onnx"])
        import tf2onnx
        import tensorflow as tf

    print(f"Loading model: {h5_path}")
    try:
        # Load model structure
        model = tf.keras.models.load_model(h5_path, compile=False)
        
        print("Converting to ONNX format...")
        # Convert!
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        
        with open(onnx_path, "wb") as f:
            f.write(model_proto.SerializeToString())
            
        print(f"\nSUCCESS! {onnx_path} created.")
        print("Now: Sync/Push this file to GitHub to finish deployment.")
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        print("Tip: If you see 'Unrecognized keyword arguments', it means your local Keras version is too new.")
        print("Try: pip install tensorflow-cpu==2.15.0 --force-reinstall")

if __name__ == "__main__":
    convert()
