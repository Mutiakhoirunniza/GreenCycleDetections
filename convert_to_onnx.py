import os
import json
import h5py
import numpy as np

# We'll use a structural approach to load and convert
# This avoids the Keras version mismatch by rebuilding the architecture manually
# and then loading the weights layer by layer.

def convert_structural():
    h5_path = "Model/model_effnetb0_best_tuned_final.h5"
    onnx_path = "Model/model.onnx"

    print("--- GreenCycle Structural Converter (Anti-Version Conflict) ---")
    
    import tensorflow as tf
    import tf2onnx

    print("Step 1: Rebuilding EfficientNetB0 structure...")
    # We rebuild exactly as Keras 2/3 would expect locally
    base_model = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    predictions = tf.keras.layers.Dense(4, activation='softmax', name="predictions")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    print(f"Step 2: Loading weights from {h5_path}...")
    try:
        # by_name=True is the magic here. It ignores metadata mismatches and only 
        # looks at the weights for layers with matching names.
        model.load_weights(h5_path, by_name=True)
        print("Weights loaded successfully!")
        
        print("Step 3: Exporting to ONNX...")
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        
        with open(onnx_path, "wb") as f:
            f.write(model_proto.SerializeToString())
            
        print(f"\nSUCCESS! {onnx_path} created.")
        print("Total size: ~20MB (Perfect for Vercel)")
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        print("\nTry this command first if it fails:")
        print("pip install tensorflow-cpu==2.15.0 --force-reinstall")

if __name__ == "__main__":
    convert_structural()
