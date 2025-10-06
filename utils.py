import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(img: Image.Image):
    SIZE = 224
    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize((SIZE, SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    return img_array