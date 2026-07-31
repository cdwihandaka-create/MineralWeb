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
    page_title="Sistem Klasifikasi Mineral",
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
<div class="overline">Analisis Spesimen Geologi</div>

<div class="main-title">
Sistem Klasifikasi Mineral
</div>

<div class="sub-title">
Identifikasi Mineral Berbasis Deep Learning · Arsitektur CNN
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
    st.metric("Kelas", "5")

with col3:
    st.metric("Ukuran Input", "224 × 224")

with col4:
    st.metric("Framework", "TensorFlow Lite")

st.divider()

# ==========================================================
# IMAGE UPLOAD
# ==========================================================

st.markdown(
    """
<div class="section-title">Unggah Gambar Mineral</div>

<p class="upload-text">
Seret & lepas gambar di sini, atau klik untuk memilih file
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
<span>Menganalisis spesimen...</span>
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

    st.markdown('<div class="section-title">Perbandingan Gambar</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="image-frame"><p class="image-label">Gambar Diunggah</p>',
            unsafe_allow_html=True
        )
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="image-frame"><p class="image-label">Gambar Referensi</p>',
            unsafe_allow_html=True
        )
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning("Gambar referensi tidak tersedia.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================
    # PREDICTION RESULT
    # ==========================================================

    st.divider()

    st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 1], gap="large")

    with left:
        st.markdown(
            f"""
<div class="prediction-card mineral-{mineral_style}">

<div class="scan"></div>

<div class="eyebrow">Hasil Klasifikasi</div>

<h2>{mineral}</h2>

<p>Hasil prediksi model CNN</p>

<hr>

<span class="confidence-badge">Tingkat keyakinan {confidence:.2f}%</span>

</div>
""",
            unsafe_allow_html=True
        )

    with right:
        st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2f}%")

        if confidence >= 95:
            status, status_class = "Sangat Yakin", "status-high"
        elif confidence >= 80:
            status, status_class = "Yakin", "status-good"
        elif confidence >= 60:
            status, status_class = "Cukup Yakin", "status-moderate"
        else:
            status, status_class = "Kurang Yakin", "status-low"

        st.markdown(
            f'<div class="status-card {status_class}">{status}</div>',
            unsafe_allow_html=True
        )

    st.progress(confidence / 100)

    # ==========================================================
    # CLASSIFICATION PROBABILITY
    # ==========================================================

    st.divider()

    st.markdown('<div class="section-title">Probabilitas Klasifikasi</div>', unsafe_allow_html=True)

    class_names = ["Azurite", "Copper", "Hematite", "Malachite", "Pyrite"]

    df = pd.DataFrame({
        "Mineral": class_names,
        "Probabilitas (%)": prediction * 100
    })

    df["Probabilitas (%)"] = df["Probabilitas (%)"].round(2)
    df = df.sort_values(by="Probabilitas (%)", ascending=False)

    st.markdown("### Distribusi Probabilitas")

    top_mineral = df.iloc[0]["Mineral"]

    for _, row in df.iterrows():

        col_left, col_right = st.columns([5, 1])
        is_top = row["Mineral"] == top_mineral
        swatch = MINERAL_STYLE.get(row["Mineral"], "")

        with col_left:
            badge = " <span class='top-badge'>Paling Cocok</span>" if is_top else ""
            st.markdown(
                f"<span class='mineral-swatch swatch-{swatch}'></span>"
                f"**{row['Mineral']}**{badge}",
                unsafe_allow_html=True
            )
            st.progress(row["Probabilitas (%)"] / 100)

        with col_right:
            st.markdown(f"<span class='data-mono'>{row['Probabilitas (%)']:.2f}%</span>", unsafe_allow_html=True)

    st.markdown("### Visualisasi")

    st.bar_chart(df.set_index("Mineral"), use_container_width=True)

    st.markdown("### Hasil Detail")

    st.dataframe(df, use_container_width=True, hide_index=True)

    # ==========================================================
    # MINERAL INFORMATION
    # ==========================================================

    st.divider()

    st.markdown('<div class="section-title">Informasi Mineral</div>', unsafe_allow_html=True)

    info = MINERAL_INFO[mineral]

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="info-card"><h3>Informasi Umum</h3>',
            unsafe_allow_html=True
        )
        st.markdown(f"**Rumus Kimia**  \n<span class='data-mono'>{info['formula']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Warna**  \n{info['color']}")
        st.markdown(f"**Tingkat Kekerasan**  \n<span class='data-mono'>{info['hardness']}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="info-card"><h3>Negara Penghasil</h3>',
            unsafe_allow_html=True
        )
        for country in info["source"]:
            st.markdown(f"• {country}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Deskripsi")

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
SISTEM KLASIFIKASI MINERAL

Tanggal Prediksi:
{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

Mineral Terprediksi:
{mineral}

Tingkat Keyakinan:
{confidence:.2f}%

Rumus Kimia:
{info['formula']}

Warna:
{info['color']}

Tingkat Kekerasan:
{info['hardness']}

Deskripsi:
{info['description']}

Negara Penghasil:
{', '.join(info['source'])}

Kegunaan:
{', '.join(info['uses'])}
"""

    st.download_button(
        label="Unduh Laporan Prediksi",
        data=result,
        file_name=f"Laporan_Prediksi_{mineral}.txt",
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
<h2>Unggah Gambar Mineral</h2>
<p>Format yang didukung: JPG · JPEG · PNG</p>
</div>
""",
        unsafe_allow_html=True
    )

st.markdown("---")

st.markdown(
    """
<div class="footer">
Sistem Klasifikasi Mineral — Universitas Gunadarma
</div>
""",
    unsafe_allow_html=True
)
