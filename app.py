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
    with open(css_path, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
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
        Convolutional Neural Network Based Mineral Identification
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ==========================================================
# UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload Mineral Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# IF IMAGE EXISTS
# ==========================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    mineral, confidence, prediction = predict_image(image)

    image_path = os.path.join(
        "assets",
        mineral.lower() + ".jpg"
    )

    # ======================================================
    # IMAGE COMPARISON
    # ======================================================

    st.markdown("## Image Comparison")

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

            st.warning("Reference image not found.")

    st.divider()

    # ======================================================
    # PREDICTION
    # ======================================================

    st.markdown("## Prediction Result")

    c1, c2 = st.columns([3, 1])

    with c1:

        st.markdown(
            f"""
            <div class="prediction-card">

            <h2>{mineral}</h2>

            <p>Predicted Mineral</p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    st.progress(confidence / 100)

    st.divider()

    # ======================================================
    # PROBABILITY
    # ======================================================

    st.markdown("## Classification Probability")

    class_names = [
        "Azurite",
        "Copper",
        "Hematite",
        "Malachite",
        "Pyrite",
    ]

    df = pd.DataFrame(
        {
            "Mineral": class_names,
            "Probability (%)": prediction * 100,
        }
    )

    st.bar_chart(
        df.set_index("Mineral"),
        use_container_width=True
    )

    st.dataframe(
        df.style.format(
            {"Probability (%)": "{:.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ======================================================
    # INFORMATION
    # ======================================================

    info = MINERAL_INFO[mineral]

    st.markdown("## Mineral Information")

    left, right = st.columns(2)

    with left:

        st.markdown("### General Information")

        st.write(
            f"**Chemical Formula:** {info['formula']}"
        )

        st.write(
            f"**Color:** {info['color']}"
        )

        st.write(
            f"**Hardness:** {info['hardness']}"
        )

    with right:

        st.markdown("### Producing Countries")

        for country in info["source"]:

            st.write("•", country)

    st.markdown("### Description")

    st.write(info["description"])

    st.markdown("### Uses")

    cols = st.columns(len(info["uses"]))

    for col, use in zip(cols, info["uses"]):

        col.success(use)

    st.divider()

    # ======================================================
    # DOWNLOAD
    # ======================================================

    result = f"""
MINERAL CLASSIFICATION SYSTEM

Date:
{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

Predicted Mineral:
{mineral}

Confidence:
{confidence:.2f}%

Chemical Formula:
{info['formula']}

Color:
{info['color']}

Hardness:
{info['hardness']}

Description:
{info['description']}

Producing Countries:
{', '.join(info['source'])}

Uses:
{', '.join(info['uses'])}
"""

    st.download_button(
        "Download Prediction Report",
        result,
        file_name=f"{mineral}.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ==========================================================
# NO IMAGE
# ==========================================================

else:

    st.markdown(
        """
        <div class="upload-box">

        <h3>Upload a mineral image to begin classification.</h3>

        Supported formats: JPG, JPEG, PNG

        </div>
        """,
        unsafe_allow_html=True,
    )
