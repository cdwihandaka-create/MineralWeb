import os
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from utils.predict import predict_v1, predict_fusion
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

st.markdown(
    """
<div class="facet-field">
<div class="aurora"></div>
<div class="glow-orb orb-azurite"></div>
<div class="glow-orb orb-gold"></div>
<div class="glow-orb orb-malachite"></div>
<span></span><span></span><span></span><span></span><span></span>
<span></span><span></span><span></span><span></span><span></span>
<span class="twinkle"></span><span class="twinkle"></span><span class="twinkle"></span>
<span class="twinkle"></span><span class="twinkle"></span><span class="twinkle"></span>
<span class="twinkle"></span><span class="twinkle"></span><span class="twinkle"></span>
<span class="twinkle"></span>
<span class="dust"></span><span class="dust"></span><span class="dust"></span>
<span class="dust"></span><span class="dust"></span><span class="dust"></span>
</div>
""",
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
# SUPPORTED MINERAL CLASSES
# ==========================================================

st.markdown('<div class="section-title">Mineral yang Dapat Diklasifikasikan</div>', unsafe_allow_html=True)

st.markdown(
    '<p class="upload-text">Sistem ini dilatih untuk mengenali 5 jenis mineral berikut</p>',
    unsafe_allow_html=True
)

class_order = ["Azurite", "Copper", "Hematite", "Malachite", "Pyrite"]
class_cols = st.columns(5)

for col, mineral_name in zip(class_cols, class_order):
    style = MINERAL_STYLE.get(mineral_name, "")
    info = MINERAL_INFO.get(mineral_name, {})
    thumb_path = os.path.join("assets", f"{mineral_name.lower()}.jpg")

    with col:
        st.markdown(f'<div class="class-card class-{style}">', unsafe_allow_html=True)

        if os.path.exists(thumb_path):
            st.image(thumb_path, use_container_width=True)

        st.markdown(
            f"""
<div class="class-card-body">
<span class="mineral-swatch swatch-{style}"></span>
<span class="class-name">{mineral_name}</span>
<div class="class-formula">{info.get('formula', '')}</div>
</div>
</div>
""",
            unsafe_allow_html=True
        )

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

    mineral, confidence, prediction = predict_v1(image)
    mineral_fusion, confidence_fusion, prediction_fusion = predict_fusion(image)

    loader.empty()

    # ==========================================================
    # TENTUKAN TIER KEYAKINAN
    # < 70%      -> tidak dikenali (unknown)
    # 70% - 80%  -> perlu verifikasi manual
    # >= 80%     -> hasil dianggap valid
    # ==========================================================

    UNKNOWN_THRESHOLD = 70
    VERIFY_THRESHOLD = 80

    if confidence < UNKNOWN_THRESHOLD:
        tier = "unknown"
    elif confidence < VERIFY_THRESHOLD:
        tier = "verify"
    else:
        tier = "confident"

    mineral_style = MINERAL_STYLE.get(mineral, "")
    image_path = os.path.join("assets", f"{mineral.lower()}.jpg")

    # ==========================================================
    # IMAGE COMPARISON
    # ==========================================================

    st.divider()

    st.markdown('<div class="section-title">Perbandingan Gambar</div>', unsafe_allow_html=True)

    if tier == "unknown":

        st.markdown(
            '<div class="image-frame"><p class="image-label">Gambar Diunggah</p>',
            unsafe_allow_html=True
        )
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption("Gambar referensi tidak ditampilkan karena hasil klasifikasi belum bisa dipastikan.")

    else:

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                '<div class="image-frame"><p class="image-label">Gambar Diunggah</p>',
                unsafe_allow_html=True
            )
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            label = "Kandidat Referensi (Belum Terverifikasi)" if tier == "verify" else "Gambar Referensi"
            st.markdown(
                f'<div class="image-frame"><p class="image-label">{label}</p>',
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

        if tier == "unknown":

            st.markdown(
                f"""
<div class="prediction-frame mineral-unknown">
<div class="prediction-card mineral-unknown">

<div class="scan"></div>

<div class="eyebrow">Hasil Klasifikasi</div>

<h2>Tidak Dikenali</h2>

<p>Gambar tidak cocok dengan kelas manapun yang dikenali sistem</p>

<hr>

<span class="confidence-badge">Keyakinan tertinggi hanya {confidence:.2f}%</span>

</div>
</div>
""",
                unsafe_allow_html=True
            )

        elif tier == "verify":

            st.markdown(
                f"""
<div class="prediction-frame mineral-verify">
<div class="prediction-card mineral-verify">

<div class="scan"></div>

<div class="eyebrow">Hasil Klasifikasi</div>

<h2>Perlu Verifikasi</h2>

<p>Kandidat tertinggi: {mineral} — belum memenuhi ambang keyakinan minimum</p>

<hr>

<span class="confidence-badge">Tingkat keyakinan {confidence:.2f}%</span>

</div>
</div>
""",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
<div class="prediction-frame mineral-{mineral_style}">
<div class="prediction-card mineral-{mineral_style}">

<div class="scan"></div>

<div class="eyebrow">Hasil Klasifikasi</div>

<h2>{mineral}</h2>

<p>Hasil prediksi model CNN</p>

<hr>

<span class="confidence-badge">Tingkat keyakinan {confidence:.2f}%</span>

</div>
</div>
""",
                unsafe_allow_html=True
            )

    with right:
        st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2f}%")

        if tier == "unknown":
            status, status_class = "Tidak Dikenali", "status-low"
        elif tier == "verify":
            status, status_class = "Perlu Verifikasi", "status-moderate"
        elif confidence >= 95:
            status, status_class = "Sangat Yakin", "status-high"
        else:
            status, status_class = "Yakin", "status-good"

        st.markdown(
            f'<div class="status-card {status_class}">{status}</div>',
            unsafe_allow_html=True
        )

    st.progress(confidence / 100)

    if tier == "unknown":
        st.markdown(
            '<div class="caveat-banner danger">'
            '<span class="icon">✕</span>'
            '<span>Sistem tidak dapat memastikan gambar ini termasuk salah satu dari 5 kelas mineral yang dikenali '
            '(Azurite, Copper, Hematite, Malachite, Pyrite). Kemungkinan gambar berasal dari luar cakupan dataset, '
            'kualitas gambar kurang jelas, atau objek bukan mineral.</span>'
            '</div>',
            unsafe_allow_html=True
        )
    elif tier == "verify":
        st.markdown(
            '<div class="caveat-banner warn">'
            '<span class="icon">!</span>'
            '<span>Tingkat keyakinan berada di zona abu-abu (70–80%). Informasi di bawah ini mengacu pada kandidat '
            'dengan probabilitas tertinggi, namun disarankan verifikasi manual sebelum dijadikan kesimpulan akhir.</span>'
            '</div>',
            unsafe_allow_html=True
        )

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
            if is_top and tier == "confident":
                badge = " <span class='top-badge'>Paling Cocok</span>"
            elif is_top:
                badge = " <span class='top-badge'>Kandidat Tertinggi</span>"
            else:
                badge = ""
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
    # PERBANDINGAN DENGAN MODEL FUSION
    # ==========================================================

    st.divider()

    st.markdown('<div class="section-title">Perbandingan dengan MineralCNN_Fusion</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="upload-text">Prediksi model kedua (CNN + fitur warna HSV + fitur tekstur LBP) sebagai pembanding</p>',
        unsafe_allow_html=True
    )

    col_v1, col_fusion = st.columns(2)

    with col_v1:
        st.markdown("#### MineralCNN_V1")
        st.metric(label="Prediksi", value=mineral)
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

    with col_fusion:
        st.markdown("#### MineralCNN_Fusion")
        st.metric(label="Prediksi", value=mineral_fusion)
        st.metric(label="Confidence", value=f"{confidence_fusion:.2f}%")

    df_fusion = pd.DataFrame({
        "Mineral": class_names,
        "Probabilitas (%)": prediction_fusion * 100
    })
    df_fusion["Probabilitas (%)"] = df_fusion["Probabilitas (%)"].round(2)
    df_fusion = df_fusion.sort_values(by="Probabilitas (%)", ascending=False)

    st.markdown("##### Distribusi Probabilitas — MineralCNN_Fusion")
    st.bar_chart(df_fusion.set_index("Mineral"), use_container_width=True)

    # ==========================================================
    # MINERAL INFORMATION
    # ==========================================================

    st.divider()

    st.markdown('<div class="section-title">Informasi Mineral</div>', unsafe_allow_html=True)

    if tier == "unknown":

        st.markdown(
            '<div class="caveat-banner danger">'
            '<span class="icon">✕</span>'
            '<span>Informasi mineral tidak ditampilkan karena hasil klasifikasi belum bisa dipastikan. '
            'Silakan coba unggah gambar lain dengan pencahayaan dan sudut yang lebih jelas.</span>'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        if tier == "verify":
            st.markdown(
                '<div class="caveat-banner warn">'
                '<span class="icon">!</span>'
                f'<span>Informasi berikut berdasarkan kandidat <strong>{mineral}</strong> dengan probabilitas '
                'tertinggi, namun belum terverifikasi sepenuhnya.</span>'
                '</div>',
                unsafe_allow_html=True
            )

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

    if tier == "unknown":

        result = f"""
SISTEM KLASIFIKASI MINERAL

Tanggal Analisis:
{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

Status:
TIDAK DIKENALI

Catatan:
Sistem tidak dapat mengklasifikasikan gambar ke salah satu dari 5 kelas
mineral yang dikenali (Azurite, Copper, Hematite, Malachite, Pyrite).

Kandidat tertinggi (referensi saja, tidak dapat dipastikan):
{top_mineral} ({df.iloc[0]['Probabilitas (%)']:.2f}%)
"""

        file_name = "Laporan_Prediksi_TidakDikenali.txt"

    else:

        info = MINERAL_INFO[mineral]

        verify_note = (
            "\nCatatan:\nHasil ini berada pada zona keyakinan 70-80% dan disarankan\n"
            "untuk diverifikasi secara manual sebelum dijadikan kesimpulan akhir.\n"
            if tier == "verify" else ""
        )

        result = f"""
SISTEM KLASIFIKASI MINERAL

Tanggal Prediksi:
{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

Mineral Terprediksi:
{mineral}

Tingkat Keyakinan:
{confidence:.2f}%
{verify_note}
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

        file_name = f"Laporan_Prediksi_{mineral}.txt"

    st.download_button(
        label="Unduh Laporan Prediksi",
        data=result,
        file_name=file_name,
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
