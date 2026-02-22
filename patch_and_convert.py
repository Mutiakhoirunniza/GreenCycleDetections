import h5py
import json
import os

h5_path = "Model/model_effnetb0_best_tuned_final.h5"
h5_patched = "Model/model_patched.h5"

print("Patching H5 file to fix Keras 3 compatibility...")
with h5py.File(h5_path, 'r') as f:
    config = f.attrs.get('model_config')
    if isinstance(config, bytes):
        config = config.decode('utf-8')
    
    # Replace batch_shape with batch_input_shape (the old name K3 expects in some paths)
    # or just remove it if it crashes.
    config = config.replace('"batch_shape"', '"batch_input_shape"')
    
    # Create new file with patched config
    with h5py.File(h5_patched, 'w') as f_new:
        for key in f.keys():
            f.copy(key, f_new)
        for attr_key in f.attrs.keys():
            if attr_key == 'model_config':
                f_new.attrs[attr_key] = config
            else:
                f_new.attrs[attr_key] = f.attrs[attr_key]

print("Patching successful! Now trying to convert patched model...")

import tensorflow as tf
try:
    model = tf.keras.models.load_model(h5_patched, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("Model/model.tflite", "wb") as f:
        f.write(tflite_model)
    print("CONVERSION SUCCESSFUL!")
except Exception as e:
    print(f"CONVERSION FAILED: {e}")
