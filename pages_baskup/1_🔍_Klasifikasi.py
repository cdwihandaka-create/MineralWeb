import streamlit as st
import pandas as pd
import os
from PIL import Image

from utils.predict import predict_image
from data.mineral_info import MINERAL_INFO

# ==========================================
# Judul
# ==========================================

st.title("🔍 Klasifikasi Batu Mineral")

st.write(
    "Upload gambar batu mineral, kemudian sistem akan mengidentifikasi jenis mineral menggunakan model Convolutional Neural Network (CNN)."
)

uploaded_file = st.file_uploader(
    "Upload Gambar Mineral",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# Prediksi
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    mineral, confidence, probability = predict_image(image)

    if mineral not in MINERAL_INFO:
        st.error(f"Mineral '{mineral}' tidak ditemukan.")
        st.stop()

    info = MINERAL_INFO[mineral]

    image_path = os.path.join(
        "assets",
        mineral.lower() + ".jpg"
    )

    # ==========================================
    # Hasil Prediksi
    # ==========================================

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            caption="Gambar yang Diunggah",
            use_container_width=True
        )

    with col2:

        st.success("### 🪨 Hasil Prediksi")

        st.metric(
            "Mineral",
            mineral
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

        if confidence >= 90:
            st.success("Prediksi sangat meyakinkan.")
        elif confidence >= 70:
            st.warning("Prediksi cukup meyakinkan.")
        else:
            st.error("Prediksi kurang meyakinkan. Disarankan menggunakan gambar yang lebih jelas.")

    st.divider()

    # ==========================================
    # Grafik Probabilitas
    # ==========================================

    st.subheader("📊 Probabilitas Setiap Kelas")

    df = pd.DataFrame({
        "Mineral": [
            "Azurite",
            "Copper",
            "Hematite",
            "Malachite",
            "Pyrite"
        ],
        "Probabilitas (%)": probability * 100
    })

    st.bar_chart(
        df.set_index("Mineral")
    )

    st.divider()

    # ==========================================
    # Informasi Mineral
    # ==========================================

    st.header("📖 Informasi Mineral")

    info_col1, info_col2 = st.columns([1, 2])

    with info_col1:

        if os.path.exists(image_path):
            st.image(
                image_path,
                caption=mineral,
                use_container_width=True
            )
        else:
            st.info("Foto mineral belum tersedia.")

    with info_col2:

        st.subheader(mineral)

        st.write(info["description"])

        st.markdown(f"**🧪 Rumus Kimia** : {info['formula']}")

        st.markdown(f"**🎨 Warna** : {info['color']}")

        st.markdown(f"**💎 Kekerasan Mohs** : {info['hardness']}")

        st.markdown("### 🌍 Negara Penghasil")

        for negara in info["source"]:
            st.write(f"• {negara}")

        st.markdown("### 🏭 Pemanfaatan")

        for manfaat in info["uses"]:
            st.write(f"• {manfaat}")