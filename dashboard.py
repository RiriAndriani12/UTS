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
# SIMULASI DAFTAR KELAS (SESUAIKAN DENGAN MODEL ANDA)
# ===========================================
# Perubahan: Daftar kelas disesuaikan dengan urutan yang BENAR dari model Anda
FOOD_CLASSES = {
    0: "Ayam Goreng",
    1: "Ayam Pop",
    2: "Daging Rendang",
    3: "Dendeng Batokok",
    4: "Gulai Ikan", 
    # Pastikan ID kelas ini (0, 1, 2, 3, 4) sudah PASTI sesuai dengan urutan kelas
    # yang digunakan saat melatih model YOLOv8 dan CNN Anda.
}
NUM_CLASSES = len(FOOD_CLASSES)
CLASS_NAMES = list(FOOD_CLASSES.values())


# ===========================================
# LOAD MODELS
# ===========================================
@st.cache_resource
def load_models():
    # Model YOLOv8 harus memiliki daftar kelas yang sama secara internal
    yolo_model = YOLO("Model/Riri Andriani_Laporan 4.pt")
    # Menonaktifkan mode compile saat memuat model Keras untuk menghindari potensi error pada Streamlit
    classifier = tf.keras.models.load_model("Model/saved_model.keras", compile=False) 
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'Riri Andriani_Laporan 4.pt' dan 'saved_model.keras' ada di folder 'Model/'. Error: {e}")
    yolo_model, classifier = None, None # Set ke None jika gagal

# ===========================================
# ESTIMASI NUTRISI (FUNGSI UTK SIMULASI DATA BERDASARKAN MAKANAN)
# ===========================================
def estimate_nutrition(food_name):
    # Logika estimasi nutrisi yang lebih spesifik berdasarkan nama makanan
    if "Ayam Goreng" in food_name:
        kalori = random.randint(350, 550)
        protein = random.uniform(25, 40)
        lemak = random.uniform(15, 35)
        karbo = random.uniform(5, 20)
    elif "Ayam Pop" in food_name:
        kalori = random.randint(300, 450)
        protein = random.uniform(20, 35)
        lemak = random.uniform(10, 25)
        karbo = random.uniform(5, 15)
    elif "Daging Rendang" in food_name:
        kalori = random.randint(400, 650)
        protein = random.uniform(30, 50)
        lemak = random.uniform(25, 45)
        karbo = random.uniform(10, 30)
    elif "Dendeng Batokok" in food_name:
        kalori = random.randint(300, 450)
        protein = random.uniform(20, 35)
        lemak = random.uniform(10, 25)
        karbo = random.uniform(5, 15)
    elif "Gulai Ikan" in food_name:
        kalori = random.randint(250, 400)
        protein = random.uniform(20, 35)
        lemak = random.uniform(10, 20)
        karbo = random.uniform(10, 25)
    else: # Default/makanan lain
        kalori = random.randint(200, 600)
        protein = random.uniform(10, 40)
        lemak = random.uniform(5, 30)
        karbo = random.uniform(20, 80)
    
    return kalori, protein, lemak, karbo

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
    ["🍛 Deteksi & Estimasi Nutrisi", "🎯 Analisis Klasifikasi"]
)

# ===========================================
# MODE A – DETEKSI MAKANAN
# ===========================================
if menu == "🍛 Deteksi & Estimasi Nutrisi":
    st.header("🍽 Deteksi Makanan & Estimasi Gizi")

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
            img_path = os.path.join(sample_dir, selected_img)
            if not os.path.exists(img_path):
                 st.error(f"Gambar contoh '{selected_img}' tidak ditemukan.")
                 st.stop()
            img = Image.open(img_path)

        img = img.convert("RGB")
        st.image(img, caption="📷 Gambar yang Diuji", use_container_width=True)

        # === BAGI LAYOUT ===
        col1, col2 = st.columns(2)

        # ==============================
        # 🔍 YOLO DETECTION
        # ==============================
        detected_food_names = []
        
        with col1:
            st.subheader("🔍 Deteksi Objek (YOLOv8)")
            if yolo_model:
                results = yolo_model(img, verbose=False)
                result_img = results[0].plot()
                st.image(result_img, caption="Hasil Deteksi YOLO", use_container_width=True)

                # Mendapatkan nama objek yang dideteksi
                for r in results[0].boxes:
                    class_id = int(r.cls.item())
                    conf = r.conf.item()
                    if class_id in FOOD_CLASSES:
                        detected_food_names.append(f"{FOOD_CLASSES[class_id]} ({conf*100:.2f}%)")
                    else:
                        detected_food_names.append(f"Objek Tak Dikenal {class_id} ({conf*100:.2f}%)")
            else:
                st.warning("Model YOLO tidak dimuat.")

        # ==============================
        # 🧠 KLASIFIKASI & NUTRISI
        # ==============================
        with col2:
            st.subheader("🧠 Klasifikasi & Estimasi Nutrisi")

            predicted_food = "Makanan Tidak Diketahui"

            if detected_food_names:
                # KASUS 1: Objek terdeteksi oleh YOLO
                st.info(f"✅ Ditemukan {len(detected_food_names)} Objek Makanan:")
                
                first_detection = detected_food_names[0].split('(')[0].strip() 
                predicted_food = first_detection
                confidence = float(detected_food_names[0].split('(')[1].strip('% )'))

                st.success(f"🍽 Makanan Utama (Dari YOLO): *{predicted_food}* ({confidence:.2f}%)")
                st.caption(f"Objek terdeteksi lainnya: {', '.join(detected_food_names[1:])}" if len(detected_food_names) > 1 else "Hanya satu objek terdeteksi.")

            elif classifier:
                # KASUS 2: Tidak ada deteksi YOLO, kembali ke klasifikasi gambar tunggal dengan CNN
                st.info("⚠️ Tidak ada objek terdeteksi oleh YOLO. Melakukan klasifikasi gambar tunggal dengan CNN.")
                
                input_shape = classifier.input_shape[1:3]
                img_resized = img.resize(input_shape)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                st.caption(f"Ukuran input model: {input_shape}")
                st.caption(f"Shape array prediksi: {img_array.shape}")

                # prediksi CNN
                try:
                    preds = classifier.predict(img_array, verbose=0)[0]
                    if len(preds) == NUM_CLASSES:
                        class_names_cnn = CLASS_NAMES
                    else:
                        class_names_cnn = [f"Makanan {i+1}" for i in range(len(preds))]

                    pred_index = np.argmax(preds)
                    predicted_food = class_names_cnn[pred_index]
                    confidence = preds[pred_index] * 100

                    st.success(f"🍽 Prediksi: *{predicted_food}* ({confidence:.2f}%)")
                except Exception as e:
                    st.error(f"Gagal melakukan prediksi dengan model CNN. Error: {e}")
                    predicted_food = "Makanan Tidak Diketahui"
                    confidence = 0.0
            
            else:
                 st.warning("Kedua model (YOLO dan CNN) gagal dimuat.")
                 predicted_food = "Makanan Tidak Diketahui"
                 
            # ========================================
            # Estimasi nutrisi berdasarkan predicted_food
            # ========================================
            if predicted_food != "Makanan Tidak Diketahui":
                kalori, protein, lemak, karbo = estimate_nutrition(predicted_food)

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
                    title=f"🍴 Komposisi Gizi Perkiraan untuk *{predicted_food}*",
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
                    title="Proporsi Nutrisi Makro (Tanpa Kalori)"
                )
                st.plotly_chart(fig_donut, use_container_width=True)


# ===========================================
# MODE B – ANALISIS MODEL
# ===========================================
elif menu == "🎯 Analisis Klasifikasi":
    st.header("🎯 Analisis Performa Klasifikasi")
    file_path = "Model/evaluasi.csv"

    if os.path.exists(file_path):
        try:
            df_eval = pd.read_csv(file_path)

            st.subheader("🎯 Akurasi Tiap Kelas")
            fig_bar = px.bar(df_eval, x="kelas", y="akurasi", color="kelas",
                             title="Akurasi Model Klasifikasi per Kelas", text_auto=".2f")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("📉 Tren Loss Selama Training")
            if "epoch" in df_eval.columns and "val_loss" in df_eval.columns:
                fig_line = px.line(df_eval, x="epoch", y="val_loss",
                                     title="Perubahan Validation Loss per Epoch", markers=True)
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Kolom 'epoch' dan 'val_loss' tidak ditemukan di CSV.")
        
        except Exception as e:
             st.error(f"Gagal membaca atau memproses evaluasi.csv. Error: {e}")

    else:
        st.warning("⚠ File **evaluasi.csv** belum tersedia di folder Model/. Upload dulu hasil evaluasi model kamu.")

# ===========================================
# FOOTER
# ===========================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 | Smart Food Vision by Riri Andriani 🍱 | YOLOv8 + TensorFlow</p>",
    unsafe_allow_html=True
)
