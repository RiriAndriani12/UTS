
# ===========================================
# dashboard_visual.py
# ===========================================
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ===========================================
# LOAD MODELS (cache biar gak reload terus)
# ===========================================
@st.cache_resource
def load_models():
    yolo_model = YOLO("Model/Riri Andriani_Laporan 4.pt")
    classifier = tf.keras.models.load_model("Model/saved_model.keras")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ===========================================
# STREAMLIT UI
# ===========================================
st.set_page_config(page_title="UTS AI Dashboard", page_icon="🤖", layout="wide")
st.title("🤖 Dashboard UTS – Deteksi & Klasifikasi Gambar")

menu = st.sidebar.selectbox("Pilih Mode:", [
    "🖼️ Analisis Gambar",
    "📊 Analisis Performa Model"
])

# ===========================================
# MODE A - ANALISIS GAMBAR
# ===========================================
if menu == "🖼️ Analisis Gambar":
    st.header("🧩 Deteksi & Klasifikasi Gambar")

    sample_images = os.listdir("sample_image")
    selected_img = st.selectbox("Pilih Gambar Contoh:", sample_images)
    uploaded_file = st.file_uploader("Atau Unggah Gambar Sendiri", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
    else:
        img = Image.open(f"sample_image/{selected_img}")

    st.image(img, caption="Gambar yang Diuji", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Deteksi Objek (YOLO)")
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi YOLO", use_container_width=True)

    with col2:
        st.subheader("🧠 Klasifikasi Gambar")
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = classifier.predict(img_array)[0]
        class_names = [f"Kelas {i+1}" for i in range(len(preds))]
        df_pred = pd.DataFrame({"Kelas": class_names, "Probabilitas": preds})

        # Donut Chart
        fig_donut = px.pie(df_pred, names="Kelas", values="Probabilitas",
                           hole=0.5, title="Distribusi Probabilitas Klasifikasi")
        st.plotly_chart(fig_donut, use_container_width=True)

        # Radar Chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=preds,
            theta=class_names,
            fill='toself',
            name='Confidence per Class'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Radar Chart Confidence Tiap Kelas"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ===========================================
# MODE B - ANALISIS PERFORMA MODEL
# ===========================================
elif menu == "📊 Analisis Performa Model":
    st.header("📈 Analisis Performa Model")

    # Contoh: file evaluasi CSV berisi kolom: kelas, akurasi, f1_score
    file_path = "Model/evaluasi.csv"
    if os.path.exists(file_path):
        df_eval = pd.read_csv(file_path)

        st.subheader("🔹 Grafik Akurasi per Kelas")
        fig_bar = px.bar(df_eval, x="kelas", y="akurasi", color="kelas",
                         title="Akurasi Tiap Kelas", text_auto=".2f")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("📉 Tren Performa Model")
        if "epoch" in df_eval.columns and "val_loss" in df_eval.columns:
            fig_line = px.line(df_eval, x="epoch", y="val_loss", title="Perubahan Validation Loss per Epoch")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Kolom 'epoch' dan 'val_loss' tidak ditemukan di CSV.")
    else:
        st.warning("⚠️ File evaluasi.csv belum tersedia di folder Model/. Upload dulu hasil evaluasi model kamu.")

# ===========================================
# FOOTER
# ===========================================
st.markdown("---")
st.markdown("**© 2025 | Dashboard UTS Riri Andriani | YOLOv8 + TensorFlow**")
