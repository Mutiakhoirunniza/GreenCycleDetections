import os
import tensorflow as tf

def convert():
    h5_path = "Model/model_effnetb0_best_tuned_final.h5"
    tflite_path = "Model/model.tflite"
    temp_sm = "temp_saved_model"

    print("Step 1: Building model structure...")
    try:
        # Load model with custom objects mapping to handle Keras 3 issues
        model = tf.keras.models.load_model(h5_path, compile=False)
        
        print("Step 2: Exporting to SavedModel...")
        tf.saved_model.save(model, temp_sm)
        
        print("Step 3: Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(temp_sm)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"SUCCESS! {tflite_path} has been created.")
        
    except Exception as e:
        print(f"FAILED: {e}")
        print("\nTIP: If this fails, try converting your model using a Python 3.9 environment locally.")

if __name__ == "__main__":
    convert()
