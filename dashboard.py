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
    # Pastikan path model ini benar
    yolo_model = YOLO("Model/Riri Andriani_Laporan 4.pt")
    classifier = tf.keras.models.load_model("Model/saved_model.keras")
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file model ada di folder 'Model/': {e}")
    st.stop() # Hentikan eksekusi jika model gagal dimuat

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

# Menggunakan st.selectbox untuk tampilan dropdown
menu = st.sidebar.selectbox(
    "📂 Pilih Mode:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Gambar & Nutrisi", "Analisis Model"]
)

# Fungsi untuk memuat gambar
def load_image_input():
    sample_dir = "Sampel Image"
    img = None
    
    if not os.path.exists(sample_dir):
        st.error(f"Folder '{sample_dir}' tidak ditemukan. Pastikan sudah ada di direktori proyek.")
        return None
        
    sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Tambahkan opsi untuk 'None' atau gambar default jika daftar kosong
    if not sample_images:
        st.warning("Folder Sampel Image kosong.")
        selected_img = None
    else:
        selected_img = st.selectbox("📸 Pilih Gambar Contoh:", sample_images)
        
    uploaded_file = st.file_uploader("📤 Atau Unggah Gambar Sendiri", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
    elif selected_img:
        img = Image.open(os.path.join(sample_dir, selected_img)).convert("RGB")
        
    if img:
        st.image(img, caption="📷 Gambar yang Diuji", use_container_width=True)
        return img
    return None

# ===========================================
# MODE 1 – DETEKSI OBJEK (YOLO)
# ===========================================
if menu == "Deteksi Objek (YOLO)":
    st.header("🔍 Deteksi Objek Makanan (YOLOv8)")
    
    img = load_image_input()

    if img:
        st.subheader("Hasil Deteksi YOLOv8")
        try:
            results = yolo_model(img, verbose=False) 
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi YOLO", use_container_width=True)
            
            labels = [yolo_model.names[int(cls)] for cls in results[0].boxes.cls]
            if labels:
                st.success(f"✅ Objek Terdeteksi: **{', '.join(set(labels))}**")
            else:
                st.info("Tidak ada objek yang terdeteksi.")

        except Exception as e:
            st.error(f"Gagal menjalankan Deteksi YOLOv8: {e}")

# ===========================================
# MODE 2 – KLASIFIKASI GAMBAR & NUTRISI
# ===========================================
elif menu == "Klasifikasi Gambar & Nutrisi":
    st.header("🧠 Klasifikasi Makanan & Estimasi Nutrisi")

    img = load_image_input()
    
    if img:
        # === BAGI LAYOUT ===
        col1, col2 = st.columns(2)

        # ==============================
        # 🧠 CNN CLASSIFICATION
        # ==============================
        with col1:
            st.subheader("🧠 Hasil Klasifikasi CNN")

            try:
                # Preprocessing untuk CNN Classifier
                input_shape = classifier.input_shape[1:3]
                img_resized = img.resize(input_shape)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # prediksi CNN
                preds = classifier.predict(img_array, verbose=False)[0]
                
                # ****** PERBAIKAN: Ganti class_names dengan nama makanan aktual ******
                # ASUMSI urutan indeks kelas Anda berdasarkan folder sampel:
                # Indeks 0: Ayam
                # Indeks 1: Daging Rendang
                # Indeks 2: Dendeng Batokok
                # Indeks 3: Gulai Ikan
                class_names = ["Ayam", "Daging Rendang", "Dendeng Batokok", "Gulai Ikan"]
                
                pred_index = np.argmax(preds)
                predicted_food = class_names[pred_index] # Sekarang menampilkan nama makanan
                confidence = preds[pred_index] * 100
                
                st.metric(label="Makanan Terprediksi", value=predicted_food)
                st.success(f"🍽 Prediksi: **{predicted_food}** ({confidence:.2f}%)")

            except Exception as e:
                st.error(f"Gagal menjalankan Klasifikasi CNN: {e}")
                predicted_food = "Unknown Food" # Fallback

        # ==============================
        # 📊 ESTIMASI NUTRISI SIMULATIF
        # ==============================
        with col2:
            st.subheader("📊 Estimasi Nutrisi (Simulatif)")

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

# ===========================================
# MODE 3 – ANALISIS MODEL (ASLI)
# ===========================================
elif menu == "Analisis Model":
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
        st.warning("⚠ File evaluasi.csv belum tersedia di folder Model/. Harap tambahkan file tersebut untuk melihat analisis.")

# ===========================================
# FOOTER
# ===========================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 | Smart Food Vision by Riri Andriani 🍱 | YOLOv8 + TensorFlow</p>",
    unsafe_allow_html=True
)
