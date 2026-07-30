import os
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from utils.predict import predict_image
from data.mineral_info import MINERAL_INFO

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Mineral Classification System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# LOAD CSS
# ==========================================================

css_path = os.path.join("styles", "style.css")

if os.path.exists(css_path):

    with open(css_path, encoding="utf-8") as css:

        st.markdown(
            f"<style>{css.read()}</style>",
            unsafe_allow_html=True
        )

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
<div class="main-title">
Mineral Classification System
</div>

<div class="sub-title">
Deep Learning Based Mineral Identification
Using Convolutional Neural Network
</div>
""",
    unsafe_allow_html=True
)

# ==========================================================
# DASHBOARD INFORMATION
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Model",
        "CNN"
    )

with col2:

    st.metric(
        "Classes",
        "5"
    )

with col3:

    st.metric(
        "Input Size",
        "224 × 224"
    )

with col4:

    st.metric(
        "Framework",
        "TensorFlow Lite"
    )

st.divider()

# ==========================================================
# IMAGE UPLOAD
# ==========================================================

st.markdown(
"""
<div class="section-title">

Upload Mineral Image

</div>
""",
unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    label="Upload an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ==========================================================
# PREDICTION
# ==========================================================

if uploaded_file is not None:

    # Load Image
    image = Image.open(uploaded_file).convert("RGB")

    # Predict
    mineral, confidence, prediction = predict_image(image)

    # Reference Image
    image_path = os.path.join(
        "assets",
        f"{mineral.lower()}.jpg"
    )

# ==========================================================
# IMAGE COMPARISON
# ==========================================================

    st.divider()

    st.markdown(
        """
<div class="section-title">

Image Comparison

</div>
""",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown("### Reference Image")

        if os.path.exists(image_path):

            st.image(
                image_path,
                use_container_width=True
            )

        else:

            st.warning("Reference image is not available.")

    # ==========================================================
    # PREDICTION RESULT
    # ==========================================================

    st.divider()

    st.markdown(
        """
<div class="section-title">
Prediction Result
</div>
""",
        unsafe_allow_html=True
    )

    left, right = st.columns([3, 1], gap="large")

    # ----------------------------------------------------------
    # Prediction Card
    # ----------------------------------------------------------

    with left:

        st.markdown(
            f"""
<div class="prediction-card">

<h2>{mineral}</h2>

<p>Predicted Mineral</p>

<hr>

<p>CNN Classification Result</p>

</div>
""",
            unsafe_allow_html=True
        )

    # ----------------------------------------------------------
    # Confidence Card
    # ----------------------------------------------------------

    with right:

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        if confidence >= 95:
            status = "Highly Confident"

        elif confidence >= 80:
            status = "Confident"

        elif confidence >= 60:
            status = "Moderate"

        else:
            status = "Low Confidence"

        st.markdown(
            f"""
<div class="status-card">
{status}
</div>
""",
            unsafe_allow_html=True
        )

    st.progress(confidence / 100)

    # ==========================================================
    # CLASSIFICATION PROBABILITY
    # ==========================================================

    st.divider()

    st.markdown(
        """
<div class="section-title">
Classification Probability
</div>
""",
        unsafe_allow_html=True
    )

    class_names = [
        "Azurite",
        "Copper",
        "Hematite",
        "Malachite",
        "Pyrite"
    ]

    df = pd.DataFrame({

        "Mineral": class_names,

        "Probability (%)": prediction * 100

    })

    df["Probability (%)"] = df["Probability (%)"].round(2)

    df = df.sort_values(
        by="Probability (%)",
        ascending=False
    )

    # ----------------------------------------------------------
    # Probability Distribution
    # ----------------------------------------------------------

    st.markdown("### Probability Distribution")

    for _, row in df.iterrows():

        col_left, col_right = st.columns([5, 1])

        with col_left:

            st.markdown(
                f"**{row['Mineral']}**"
            )

            st.progress(
                row["Probability (%)"] / 100
            )

        with col_right:

            st.markdown(
                f"**{row['Probability (%)']:.2f}%**"
            )

    # ----------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------

    st.markdown("### Visualization")

    st.bar_chart(
        df.set_index("Mineral"),
        use_container_width=True
    )

    # ----------------------------------------------------------
    # Detailed Result
    # ----------------------------------------------------------

    st.markdown("### Detailed Result")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
