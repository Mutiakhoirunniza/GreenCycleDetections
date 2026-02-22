import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

def create_and_convert():
    print("Building model structure (EfficientNetB0)...")
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    h5_weights = "Model/model_effnetb0_best_tuned_final.h5"
    print(f"Loading weights from {h5_weights}...")
    try:
        model.load_weights(h5_weights)
        print("Weights loaded successfully!")
        
        print("Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open("Model/model.tflite", "wb") as f:
            f.write(tflite_model)
        print("DONE! model.tflite created in Model folder.")
    except Exception as e:
        print(f"FAILED to convert: {e}")

if __name__ == "__main__":
    create_and_convert()
