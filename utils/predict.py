import numpy as np
from PIL import Image
import tensorflow as tf
import os

CLASS_NAMES = [
    "Azurite",
    "Copper",
    "Hematite",
    "Malachite",
    "Pyrite"
]

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "MineralCNN.tflite"
)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(image):
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_idx = np.argmax(prediction)
    mineral = CLASS_NAMES[pred_idx]
    confidence = float(prediction[pred_idx] * 100)
    return mineral, confidence, prediction
