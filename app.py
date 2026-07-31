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
# MINERAL COLOR MAP
# ==========================================================

MINERAL_STYLE = {
    "Azurite": "azurite",
    "Copper": "copper",
    "Hematite": "hematite",
    "Malachite": "malachite",
    "Pyrite": "pyrite",
}

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
Deep Learning Based Mineral Identification · CNN Architecture
</div>

<div class="class-dots">
<span></span><span></span><span></span><span></span><span></span>
</div>
""",
    unsafe_allow_html=True
)

# ==========================================================
# DASHBOARD INFORMATION
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", "CNN")

with col2:
    st.metric("Classes", "5")

with col3:
    st.metric("Input Size", "224 × 224")

with col4:
    st.metric("Framework", "TensorFlow Lite")

st.divider()

# ==========================================================
# IMAGE UPLOAD
# ==========================================================

st.markdown(
    """
<div class="section-title">Upload Mineral Image</div>

<p class="upload-text">
Drag & Drop an image here or click to browse
</p>
""",
    unsafe_allow_html=True
)

st.markdown('<div class="upload-container">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# PREDICTION
# ==========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    loader = st.empty()

    loader.markdown(
        """
<div class="mineral-loader">
<div class="scan-ring"></div>
<span>Scanning specimen...</span>
</div>
""",
        unsafe_allow_html=True
    )

    mineral, confidence, prediction = predict_image(image)

    loader.empty()

    mineral_style = MINERAL_STYLE.get(mineral, "")

    image_path = os.path.join("assets", f"{mineral.lower()}.jpg")

    # ==========================================================
    # IMAGE COMPARISON
    # ==========================================================

    st.divider()

    st.markdown('<div class="section-title">Image Comparison</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="image-frame"><p class="image-label">Uploaded Specimen</p>',
            unsafe_allow_html=True
        )
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="image-frame"><p class="image-label">Reference Specimen</p>',
            unsafe_allow_html=True
        )
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning("Reference image is not available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # PREDICTION RESULT
    # ==========================================================

    st.divider()

    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 1], gap="large")

    with left:
        st.markdown(
            f"""
<div class="prediction-card mineral-{mineral_style}">

<div class="scan"></div>

<div class="eyebrow">Classification Output</div>

<h2>{mineral}</h2>

<p>CNN model prediction result</p>

<hr>

<span class="confidence-badge">{confidence:.2f}% confidence</span>

</div>
""",
            unsafe_allow_html=True
        )

    with right:
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

        if confidence >= 95:
            status, status_class = "Highly Confident", "status-high"
        elif confidence >= 80:
            status, status_class = "Confident", "status-good"
        elif confidence >= 60:
            status, status_class = "Moderate", "status-moderate"
        else:
            status, status_class = "Low Confidence", "status-low"

        st.markdown(
            f'<div class="status-card {status_class}">{status}</div>',
            unsafe_allow_html=True
        )

    st.progress(confidence / 100)

    # ==========================================================
    # CLASSIFICATION PROBABILITY
    # ==========================================================

    st.divider()

    st.markdown('<div class="section-title">Classification Probability</div>', unsafe_allow_html=True)

    class_names = ["Azurite", "Copper", "Hematite", "Malachite", "Pyrite"]

    df = pd.DataFrame({
        "Mineral": class_names,
        "Probability (%)": prediction * 100
    })

    df["Probability (%)"] = df["Probability (%)"].round(2)
    df = df.sort_values(by="Probability (%)", ascending=False)

    st.markdown("### Probability Distribution")

    top_mineral = df.iloc[0]["Mineral"]

    for _, row in df.iterrows():

        col_left, col_right = st.columns([5, 1])
        is_top = row["Mineral"] == top_mineral
        swatch = MINERAL_STYLE.get(row["Mineral"], "")

        with col_left:
            badge = " <span class='top-badge'>Top Match</span>" if is_top else ""
            st.markdown(
                f"<span class='mineral-swatch swatch-{swatch}'></span>"
                f"**{row['Mineral']}**{badge}",
                unsafe_allow_html=True
            )
            st.progress(row["Probability (%)"] / 100)

        with col_right:
            st.markdown(f"<span class='data-mono'>{row['Probability (%)']:.2f}%</span>", unsafe_allow_html=True)

    st.markdown("### Visualization")

    st.bar_chart(df.set_index("Mineral"), use_container_width=True)

    st.markdown("### Detailed Result")

    st.dataframe(df, use_container_width=True, hide_index=True)

    # ==========================================================
    # MINERAL INFORMATION
    # ==========================================================

    st.divider()

    st.markdown('<div class="section-title">Mineral Information</div>', unsafe_allow_html=True)

    info = MINERAL_INFO[mineral]

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="info-card"><h3>General Information</h3>',
            unsafe_allow_html=True
        )
        st.markdown(f"**Chemical Formula**  \n<span class='data-mono'>{info['formula']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Color**  \n{info['color']}")
        st.markdown(f"**Hardness**  \n<span class='data-mono'>{info['hardness']}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="info-card"><h3>Producing Countries</h3>',
            unsafe_allow_html=True
        )
        for country in info["source"]:
            st.markdown(f"• {country}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Description")

    st.markdown(
        f'<div class="description-card">{info["description"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Berguna untuk")

    use_cols = st.columns(len(info["uses"]))

    for col, item in zip(use_cols, info["uses"]):
        with col:
            st.markdown(f'<div class="use-card">{item}</div>', unsafe_allow_html=True)

    # ==========================================================
    # DOWNLOAD REPORT
    # ==========================================================

    st.divider()

    result = f"""
MINERAL CLASSIFICATION SYSTEM

Prediction Date:
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
        label="Download Prediction Report",
        data=result,
        file_name=f"Prediction_Report_{mineral}.txt",
        mime="text/plain",
        use_container_width=True
    )

# ==========================================================
# NO IMAGE
# ==========================================================

else:

    st.markdown(
        """
<div class="upload-box">
<div class="crystal"></div>
<h2>Upload a Mineral Image</h2>
<p>Supported formats: JPG · JPEG · PNG</p>
</div>
""",
        unsafe_allow_html=True
    )

st.markdown("---")

st.markdown(
    """
<div class="footer">
Mineral Classification System — Universitas Gunadarma
</div>
""",
    unsafe_allow_html=True
)
