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
# SIMULASI DAFTAR KELAS
# ===========================================
# ===========================================
# SIMULASI DAFTAR KELAS (PENTING: URUTAN KELAS DITUKAR SEPENUHNYA)
# ===========================================
FOOD_CLASSES = {
    0: "Ayam Goreng",
    1: "Ayam Pop",
    2: "Daging Rendang",
    3: "Dendeng Batokok",
    4: "Gulai Ikan", 
}
NUM_CLASSES = len(FOOD_CLASSES)
CLASS_NAMES = list(FOOD_CLASSES.values())


# ===========================================
# LOAD MODELS
# ===========================================
@st.cache_resource
def load_models():
    yolo_model = YOLO("Model/Riri Andriani_Laporan 4.pt")
    classifier = tf.keras.models.load_model("Model/saved_model.keras", compile=False) 
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'Riri Andriani_Laporan 4.pt' dan 'saved_model.keras' ada di folder 'Model/'. Error: {e}")
    yolo_model, classifier = None, None 

# ===========================================
# ESTIMASI NUTRISI (FUNGSI UTK SIMULASI DATA BERDASARKAN MAKANAN)
# ===========================================
# Fungsi ini HANYA akan dipanggil di Mode YOLO
def estimate_nutrition(food_name):
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
    else: 
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

# Fungsi untuk memuat gambar
def load_image_selection():
    sample_dir = "Sampel Image"
    if not os.path.exists(sample_dir):
        st.error(f"Folder '{sample_dir}' tidak ditemukan. Pastikan sudah ada di direktori proyek.")
        return None

    sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_img = st.selectbox("📸 Pilih Gambar Contoh:", sample_images)
    uploaded_file = st.file_uploader("📤 Atau Unggah Gambar Sendiri", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
    else:
        img_path = os.path.join(sample_dir, selected_img)
        if not os.path.exists(img_path):
             st.error(f"Gambar contoh '{selected_img}' tidak ditemukan.")
             return None
        img = Image.open(img_path)

    return img.convert("RGB")

# Mode di Sidebar: Deteksi YOLO (dengan Nutrisi) dan Klasifikasi CNN (murni)
menu = st.sidebar.radio(
    "📂 Pilih Mode:",
    ["🔍 Deteksi & Nutrisi YOLO", "🧠 Klasifikasi Gambar CNN"]
)

# ===========================================
# MODE 1 – DETEKSI & NUTRISI YOLO
# ===========================================
if menu == "🔍 Deteksi & Nutrisi YOLO":
    st.header("🔍 Deteksi Makanan & Estimasi Gizi (YOLOv8)")
    
    img = load_image_selection()
    if img is None:
        st.stop()

    st.image(img, caption="📷 Gambar yang Diuji", use_container_width=True)
    
    # === Inisialisasi Deteksi ===
    detected_food_names = []
    first_detected_food_name = "Makanan Tidak Diketahui"
    first_detected_confidence = 0.0
    
    # === BAGI LAYOUT ===
    col1, col2 = st.columns(2)

    # ==============================
    # 🔍 YOLO DETECTION (Kolom Kiri)
    # ==============================
    with col1:
        st.subheader("🔍 Deteksi Objek (YOLOv8)")
        if yolo_model:
            results = yolo_model(img, verbose=False)
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi YOLO (Output Visual)", use_container_width=True)

            # Mendapatkan nama objek yang dideteksi
            for i, r in enumerate(results[0].boxes):
                class_id = int(r.cls.item())
                conf = r.conf.item()
                
                food_name_by_id = FOOD_CLASSES.get(class_id, f"Objek Tak Dikenal {class_id}")
                
                # Mengambil deteksi terbaik sebagai prediksi utama
                if i == 0:
                    first_detected_food_name = food_name_by_id
                    first_detected_confidence = conf * 100
                
                detected_food_names.append(f"• {food_name_by_id} ({conf*100:.2f}%)")
        else:
            st.warning("Model YOLO tidak dimuat.")


    # ==============================
    # 📝 ESTIMASI NUTRISI (Kolom Kanan)
    # ==============================
    with col2:
        st.subheader("📝 Estimasi Nutrisi (Berdasarkan Deteksi YOLO)")

        predicted_food = first_detected_food_name

        if predicted_food != "Makanan Tidak Diketahui" and first_detected_confidence > 0:
            
            st.success(f"🍽 Makanan Utama Dideteksi: *{predicted_food}* ({first_detected_confidence:.2f}%)")

            # Menampilkan daftar deteksi
            st.markdown("##### 🎯 Objek Terdeteksi:")
            st.markdown("\n".join(detected_food_names))
            
            # --- Perhitungan Nutrisi ---
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

        else:
            st.warning("Tidak ada objek makanan yang terdeteksi atau model YOLO gagal dimuat untuk mengestimasi nutrisi.")


# ===========================================
# MODE 2 – KLASIFIKASI GAMBAR CNN (MURNI)
# ===========================================
elif menu == "🧠 Klasifikasi Gambar CNN":
    st.header("🧠 Klasifikasi Gambar (Model CNN)")

    img = load_image_selection()
    if img is None:
        st.stop()

    # Layout untuk menampilkan gambar dan hasil
    col1, col2 = st.columns(2)
    
    with col1:
        # Menampilkan gambar input di kolom kiri
        st.image(img, caption="📷 Gambar yang Diuji (Input Model Klasifikasi)", use_container_width=True)

    with col2:
        st.subheader("Hasil Prediksi Kelas")

        if classifier:
            input_shape = classifier.input_shape[1:3]
            img_resized = img.resize(input_shape)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            try:
                preds = classifier.predict(img_array, verbose=0)[0]
                
                pred_index = np.argmax(preds)
                confidence_cnn = preds[pred_index] * 100
                
                # Mengambil nama kelas yang benar dari FOOD_CLASSES
                if pred_index in FOOD_CLASSES:
                    predicted_food_cnn = FOOD_CLASSES[pred_index]
                else:
                    predicted_food_cnn = f"Makanan (ID {pred_index})"

                st.success(f"🧠 Prediksi Tertinggi: *{predicted_food_cnn}* ({confidence_cnn:.2f}%)")
                
                # PERBAIKAN: Mengubah top_k menjadi NUM_CLASSES (5)
                top_k = NUM_CLASSES 
                top_indices = np.argsort(preds)[::-1][:top_k]
                
                st.markdown("##### Probabilitas Lengkap:")
                df_preds = pd.DataFrame({
                    "Kelas": [FOOD_CLASSES.get(i, f"ID {i}") for i in top_indices],
                    "Probabilitas (%)": [preds[i] * 100 for i in top_indices]
                })
                st.dataframe(df_preds.style.format({'Probabilitas (%)': '{:.2f}%'}), use_container_width=True)

            except Exception as e:
                st.error(f"Gagal melakukan prediksi dengan model CNN. Error: {e}")
        else:
            st.warning("Model Klasifikasi CNN tidak dimuat. Prediksi tidak dapat dilakukan.")
            
# ===========================================
# FOOTER
# ===========================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 | Smart Food Vision by Riri Andriani 🍱 | YOLOv8 + TensorFlow</p>",
    unsafe_allow_html=True
)
