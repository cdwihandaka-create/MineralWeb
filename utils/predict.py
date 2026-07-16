import tensorflow as tf
import numpy as np

# ==========================================
# Nama Kelas
# ==========================================

CLASS_NAMES = [
    "Azurite",
    "Copper",
    "Hematite",
    "Malachite",
    "Pyrite"
]

# ==========================================
# Load Model
# ==========================================

import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "MineralCNN_V1.keras"
)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# ==========================================
# Prediksi Gambar
# ==========================================

def predict_image(image):

    # Resize sesuai input model
    img = image.resize((224, 224))

    # PIL -> NumPy
    img = np.array(img, dtype=np.float32)

    # Normalisasi
    img = img / 255.0

    # Tambahkan dimensi batch
    img = np.expand_dims(img, axis=0)

    # Prediksi
    prediction = model.predict(img, verbose=0)[0]

    # Ambil kelas dengan probabilitas terbesar
    pred_idx = np.argmax(prediction)

    mineral = CLASS_NAMES[pred_idx]

    confidence = float(prediction[pred_idx] * 100)

    return mineral, confidence, prediction