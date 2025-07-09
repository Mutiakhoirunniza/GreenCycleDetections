import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

def prepare_image(img_pil, target_size=(224, 224)):
    img = img_pil.resize(target_size).convert("RGB")
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def calculate_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-6))

def get_prediction_stats(probs):
    confidence = np.max(probs)
    sorted_probs = np.sort(probs)[-2:]
    gap = sorted_probs[1] - sorted_probs[0]
    return confidence, gap

def get_mean_rgb(img_array):
    return np.mean(img_array, axis=(0, 1))
