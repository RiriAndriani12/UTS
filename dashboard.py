import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import pandas as pd
import plotly.express as px
import random

# ===========================================
# LOAD MODELS
# ===========================================
@st.cache_resource
def load_models():
    yolo_model = YOLO("Model/Riri Andriani_Laporan 4.pt")
    classifier = tf.keras.models.load_model("Model/saved_model.keras")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ===========================================
# STREAMLIT CONFIG
# ===========================================
st.set_page_config(page_title="Smart Food Vision 🍱", page_icon="🍱", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #fafafa;
    }
    h1, h2, h3 {
        text-align: center;
        color: #2b2b2b;
    }
    .stButton>button {
        background-color: #ffb347;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff944d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍱 *Smart Food Vision*")
st.markdown("### AI-powered food detection and nutrition estimation")

menu = st.sidebar.radio(
    "📂 Pilih Mode:",
    ["🍛 Deteksi & Estimasi Nutrisi", "📊 Analisis Model"]
)

# ===========================================
# MODE A – DETEKSI MAKANAN
# ===========================================
if menu == "🍛 Deteksi & Estimasi Nutrisi":
    st.header("🍽 Deteksi Makanan & Estimasi Kalori")

    sample_dir = "Sampel Image"
    if not os.path.exists(sample_dir):
        st.error(f"Folder '{sample_dir}' tidak ditemukan. Pastikan sudah ada di direktori proyek.")
    else:
        sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        selected_img = st.selectbox("📸 Pilih Gambar Contoh:", sample_images)
        uploaded_file = st.file_uploader("📤 Atau Unggah Gambar Sendiri", type=["jpg", "jpeg", "png"])

        # === LOAD GAMBAR ===
        if uploaded_file:
            img = Image.open(uploaded_file)
        else:
            img = Image.open(os.path.join(sample_dir, selected_img))

        img = img.convert("RGB")  # ⬅ tambahkan agar tidak error channel
        st.image(img, caption="📷 Gambar yang Diuji", use_container_width=True)

        # === BAGI LAYOUT ===
        col1, col2 = st.columns(2)

        # ==============================
        # 🔍 YOLO DETECTION
        # ==============================
        with col1:
            st.subheader("🔍 Deteksi Objek (YOLOv8)")
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi YOLO", use_container_width=True)

        # ==============================
        # 🧠 CNN CLASSIFICATION + NUTRISI
        # ==============================
        with col2:
            st.subheader("🧠 Klasifikasi & Estimasi Nutrisi")

            input_shape = classifier.input_shape[1:3]
            img_resized = img.resize(input_shape)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # tampilkan ukuran
            st.caption(f"Ukuran input model: {input_shape}")
            st.caption(f"Shape array prediksi: {img_array.shape}")

            # prediksi CNN
            preds = classifier.predict(img_array)[0]
            class_names = [f"Makanan {i+1}" for i in range(len(preds))]
            pred_index = np.argmax(preds)
            predicted_food = class_names[pred_index]
            confidence = preds[pred_index] * 100

            st.success(f"🍽 Prediksi: *{predicted_food}* ({confidence:.2f}%)")

            # Estimasi nutrisi simulatif
            kalori = random.randint(200, 600)
            protein = random.uniform(10, 40)
            lemak = random.uniform(5, 30)
            karbo = random.uniform(20, 80)

            df_nutrisi = pd.DataFrame({
                "Nutrisi": ["Kalori (kcal)", "Protein (g)", "Lemak (g)", "Karbohidrat (g)"],
                "Nilai": [kalori, protein, lemak, karbo]
            })

            # Grafik batang
            fig_bar = px.bar(
                df_nutrisi,
                x="Nutrisi",
                y="Nilai",
                color="Nutrisi",
                title=f"🍴 Komposisi Gizi Perkiraan untuk {predicted_food}",
                text_auto=".2f",
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Grafik donat
            fig_donut = px.pie(
                df_nutrisi.iloc[1:], 
                names="Nutrisi", 
                values="Nilai",
                hole=0.5, 
                title="Proporsi Nutrisi (Tanpa Kalori)"
            )
            st.plotly_chart(fig_donut, use_container_width=True)

# ===========================================
# MODE B – ANALISIS MODEL
# ===========================================
elif menu == "📊 Analisis Model":
    st.header("📈 Analisis Performa Model")
    file_path = "Model/evaluasi.csv"

    if os.path.exists(file_path):
        df_eval = pd.read_csv(file_path)

        st.subheader("🎯 Akurasi Tiap Kelas")
        fig_bar = px.bar(df_eval, x="kelas", y="akurasi", color="kelas",
                         title="Akurasi Model per Kelas", text_auto=".2f")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("📉 Tren Loss Selama Training")
        if "epoch" in df_eval.columns and "val_loss" in df_eval.columns:
            fig_line = px.line(df_eval, x="epoch", y="val_loss",
                               title="Perubahan Validation Loss per Epoch", markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Kolom 'epoch' dan 'val_loss' tidak ditemukan di CSV.")
    else:
        st.warning("⚠ File evaluasi.csv belum tersedia di folder Model/. Upload dulu hasil evaluasi model kamu.")

# ===========================================
# FOOTER
# ===========================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 | Smart Food Vision by Riri Andriani 🍱 | YOLOv8 + TensorFlow</p>",
    unsafe_allow_html=True
)
