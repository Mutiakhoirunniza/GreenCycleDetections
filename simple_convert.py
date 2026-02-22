import h5py
import json
import os
import tensorflow as tf

def convert():
    h5_path = "Model/model_effnetb0_best_tuned_final.h5"
    tflite_path = "Model/model.tflite"

    print("Attempting to convert model using legacy loader...")
    # This often bypasses the BATCH_SHAPE error in K3
    try:
        model = tf.keras.models.load_model(h5_path, compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Quantize to save space
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"SUCCESS! Created {tflite_path}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    convert()
