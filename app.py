import os
from PIL import Image
import pandas as pd

import streamlit as st

from utils.predict import predict_image
from data.mineral_info import MINERAL_INFO
from datetime import datetime

# ==========================================================
# Konfigurasi Halaman
# ==========================================================

st.set_page_config(
    page_title="Mineral Classification System",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# CSS
# ==========================================================

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# Header
# ==========================================================

st.title("🪨 Mineral Classification System")

st.markdown("""
### Klasifikasi Batu Mineral Menggunakan CNN From Scratch

Sistem ini mampu mengklasifikasikan lima jenis batu mineral menggunakan model
Convolutional Neural Network (CNN) yang dibangun dari awal.
""")

st.divider()

# ==========================================================
# Upload
# ==========================================================

uploaded_file = st.file_uploader(
    "📤 Upload gambar mineral",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# Prediksi
# ==========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    mineral, confidence, prediction = predict_image(image)

    image_path = os.path.join(
        "assets",
        f"{mineral.lower()}.jpg"
    )

    # ======================================================
    # Perbandingan Gambar
    # ======================================================

    st.subheader("🖼️ Perbandingan Gambar")

    img1, img2 = st.columns(2)

    with img1:

        st.markdown("#### 📤 Gambar Upload")

        st.image(
            image,
            use_container_width=True
        )

    with img2:

        st.markdown(f"#### 📚 Referensi {mineral}")

        if os.path.exists(image_path):

            st.image(
                image_path,
                use_container_width=True
            )

        else:

            st.warning("Gambar referensi belum tersedia.")

    st.divider()

    # ======================================================
    # Hasil Prediksi
    # ======================================================

    st.subheader("🔍 Hasil Prediksi")

    st.success(f"## 🪨 {mineral}")

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(confidence / 100)

    # ======================================================
    # Visualisasi Confidence Semua Kelas
    # ======================================================

    st.subheader("📊 Probabilitas Semua Kelas")

    class_names = [
        "Azurite",
        "Copper",
        "Hematite",
        "Malachite",
        "Pyrite"
    ]

    df = pd.DataFrame({
        "Mineral": class_names,
        "Confidence (%)": prediction * 100
    })

    st.bar_chart(
        df.set_index("Mineral")
    )

    st.dataframe(
        df.style.format({
            "Confidence (%)": "{:.2f}"
        }),
        use_container_width=True
    )

    st.divider()

    # ======================================================
    # Informasi Mineral
    # ======================================================

    info = MINERAL_INFO[mineral]

    st.subheader("📖 Informasi Mineral")

    info1, info2 = st.columns(2)

    with info1:

        st.write(f"**Formula Kimia** : {info['formula']}")
        st.write(f"**Warna** : {info['color']}")
        st.write(f"**Kekerasan** : {info['hardness']}")

    with info2:

        st.markdown("#### 🌍 Negara Penghasil")

        for negara in info["source"]:
            st.markdown(f"- {negara}")

    st.markdown("#### 📝 Deskripsi")

    st.write(info["description"])

    st.markdown("#### ⚙️ Pemanfaatan")

    cols = st.columns(len(info["uses"]))

    for col, penggunaan in zip(cols, info["uses"]):
        col.info(penggunaan)

    # ======================================================
    # Download Hasil Prediksi
    # ======================================================

    st.divider()

    st.subheader("💾 Download Hasil Prediksi")

    hasil = f"""
=========================================
MINERAL CLASSIFICATION SYSTEM
=========================================

Tanggal Prediksi : {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}
Mineral          : {mineral}
Confidence       : {confidence:.2f} %

-----------------------------------------
INFORMASI MINERAL
-----------------------------------------

Formula Kimia :
{info['formula']}

Warna :
{info['color']}

Kekerasan :
{info['hardness']}

Deskripsi :
{info['description']}

Negara Penghasil :
{', '.join(info['source'])}

Pemanfaatan :
{', '.join(info['uses'])}
"""

    st.download_button(
        label="⬇ Download Hasil Prediksi (.txt)",
        data=hasil,
        file_name=f"Hasil_Prediksi_{mineral}.txt",
        mime="text/plain",
        use_container_width=True
    )

else:

    st.info("Silakan upload gambar mineral untuk memulai klasifikasi.")