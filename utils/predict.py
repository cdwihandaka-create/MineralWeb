import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import os

CLASS_NAMES = [
    "Azurite",
    "Copper",
    "Hematite",
    "Malachite",
    "Pyrite"
]

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ============================================================
# MODEL V1 (Keras)
# ============================================================
V1_MODEL_PATH = os.path.join(BASE_DIR, "MineralCNN_V1.keras")
model_v1 = tf.keras.models.load_model(V1_MODEL_PATH)

# ============================================================
# MODEL FUSION (Keras)
# ============================================================
FUSION_MODEL_PATH = os.path.join(BASE_DIR, "MineralCNN_Fusion.keras")
model_fusion = tf.keras.models.load_model(FUSION_MODEL_PATH)

# ============================================================
# EKSTRAKSI FITUR HANDCRAFTED (sama seperti di notebook)
# ============================================================
def extract_hsv_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    feature = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
    feature = feature / np.sum(feature)
    return feature

def extract_lbp_features(image):
    gray = rgb2gray(image)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-8)
    return hist

# ============================================================
# PREPROCESSING CITRA (dipakai kedua model)
# ============================================================
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    return img

# ============================================================
# PREDIKSI - MineralCNN_V1 (Keras)
# ============================================================
def predict_v1(image):
    img = preprocess_image(image)
    img_batch = np.expand_dims(img, axis=0)
    prediction = model_v1.predict(img_batch, verbose=0)[0]
    pred_idx = np.argmax(prediction)
    mineral = CLASS_NAMES[pred_idx]
    confidence = float(prediction[pred_idx] * 100)
    return mineral, confidence, prediction

# ============================================================
# PREDIKSI - MineralCNN_Fusion (Keras)
# ============================================================
def predict_fusion(image):
    img = preprocess_image(image)
    img_uint8 = (img * 255).astype(np.uint8)
    hsv_feat = extract_hsv_features(img_uint8)
    lbp_feat = extract_lbp_features(img_uint8)

    img_batch = np.expand_dims(img, axis=0)
    hsv_batch = np.expand_dims(hsv_feat, axis=0)
    lbp_batch = np.expand_dims(lbp_feat, axis=0)

    prediction = model_fusion.predict(
        [img_batch, hsv_batch, lbp_batch],
        verbose=0
    )[0]
    pred_idx = np.argmax(prediction)
    mineral = CLASS_NAMES[pred_idx]
    confidence = float(prediction[pred_idx] * 100)
    return mineral, confidence, prediction

# Alias biar kompatibel kalau ada bagian lain yang masih manggil predict_image()
predict_image = predict_v1
