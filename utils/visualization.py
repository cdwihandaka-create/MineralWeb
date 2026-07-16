import cv2
import numpy as np


def create_visualizations(image):
    """
    Membuat visualisasi citra:
    - RGB
    - Grayscale
    - HSV
    - Edge Detection (Canny)
    """

    img = np.array(image)

    # RGB
    rgb = img.copy()

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Edge Detection
    edges = cv2.Canny(gray, 100, 200)

    return {
        "RGB": rgb,
        "Grayscale": gray,
        "HSV": hsv,
        "Edges": edges
    }