# ===========================================
# 🔐 NONAKTIFKAN oneDNN SEBELUM TENSORFLOW
# ===========================================
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ===========================================
# 📦 IMPORT LIBRARY
# ===========================================
import streamlit as st
import numpy as np
import pandas as pd
import base64
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from io import BytesIO
import cv2

# ===========================================
# ⚙️ KONFIGURASI STREAMLIT
# ===========================================
st.set_page_config(page_title="GreenCycleDetections", layout="centered")

# ===========================================
# 🎨 LOAD CUSTOM CSS
# ===========================================
if os.path.exists("style.css"):
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===========================================
# 🔍 LOAD MODEL DAN LABEL
# ===========================================

model_path = os.path.join("Model", "best_model.h5")
model = load_model(model_path)

class_labels = ["kaca", "kertas", "logam", "plastik"]

penjelasan = {
    "kaca": "merupakan bahan anorganik yang dapat didaur ulang tanpa mengurangi kualitasnya.",
    "kertas": "merupakan sampah yang berasal dari bahan selulosa dan sering kali tercemar.",
    "logam": "seperti kaleng dapat didaur ulang dan digunakan kembali dalam industri.",
    "plastik": "sulit terurai dan perlu pengolahan khusus untuk didaur ulang atau dimanfaatkan ulang."
}

waktu_daur_ulang = {
    "kaca": "❗ Tidak terurai secara alami, tapi 100% dapat didaur ulang berulang kali.",
    "kertas": "♻️ Sekitar 2–6 minggu jika tidak tercemar.",
    "logam": "🔁 Sekitar 50 tahun, tapi bisa didaur ulang berkali-kali.",
    "plastik": "⚠️ Bisa mencapai 100–1000 tahun tergantung jenisnya."
}

# ===========================================
# 🧠 DETEKSI WAJAH
# ===========================================
def contains_face(img_pil):
    img_cv = np.array(img_pil.convert("RGB"))
    img_cv = cv2.resize(img_cv, (300, 300))  # resize biar wajahnya masuk skala ideal
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50))
    return len(faces) > 0

# ===========================================
# 🔧 FUNGSI UTILITAS
# ===========================================
def convert_image_to_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def predict_image(img, threshold=0.8, entropy_threshold=1.0):
    if contains_face(img):
        return "manusia", 0.0, [], img, 0.0, 0.0, np.zeros((224, 224, 3))
    
    img = img.resize((224, 224)).convert("RGB")
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred_probs = model.predict(img_array)[0]
    confidence = np.max(pred_probs)
    entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-6))
    sorted_probs = np.sort(pred_probs)[-2:]
    gap = sorted_probs[1] - sorted_probs[0]

    if (confidence < threshold or entropy > entropy_threshold or gap < 0.10) and not st.session_state.force_classify:
        label = "unknown"
    else:
        label = class_labels[np.argmax(pred_probs)]

    return label, confidence, pred_probs, img, entropy, gap, img_array[0]

# ===========================================
# ➕ INPUT DARI KAMERA
# ===========================================
def capture_from_webcam_streamlit():
    camera_image = st.camera_input("📷 Ambil Foto dari Kamera")
    if camera_image:
        return Image.open(camera_image)
    return None

# ===========================================
# 🔄 SESSION STATE
# ===========================================
for key in ["show_input_box", "img", "show_edu", "force_classify"]:
    if key not in st.session_state:
        st.session_state[key] = False if key != "img" else None

# ===========================================
# 📘 HALAMAN EDUKASI
# ===========================================
if st.session_state.show_edu:
    if os.path.exists("educational_text.md"):
        st.markdown(open("educational_text.md", encoding="utf-8").read())
    if st.button("🕙 Kembali ke Klasifikasi"):
        st.session_state.show_edu = False
        st.session_state.show_input_box = True
        st.rerun()
    st.stop()

# ===========================================
# 🏠 HALAMAN UTAMA
# ===========================================
if os.path.exists("Images/ilustrasi-lingkungan.jpg"):
    st.image("Images/ilustrasi-lingkungan.jpg", use_container_width=True)

st.markdown("<h1>GreenCycleDetections</h1><h4>Klasifikasi Otomatis Sampah Anorganik Berbasis Citra Digital</h4>", unsafe_allow_html=True)

# ===========================================
# 📄 INPUT GAMBAR
# ===========================================
if not st.session_state.img:
    if st.button("♻️ Start Classification", use_container_width=True):
        st.session_state.show_input_box = True

    if st.session_state.show_input_box:
        st.markdown("<div class='waste-prompt'>Pilih jenis sampah anorganik yang ingin Anda klasifikasikan</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                st.session_state.img = Image.open(uploaded_file).convert("RGB")
                st.session_state.show_input_box = False
                st.rerun()

        with col2:
            webcam_image = capture_from_webcam_streamlit()
            if webcam_image:
                st.session_state.img = webcam_image.convert("RGB")
                st.session_state.show_input_box = False
                st.rerun()

# ===========================================
# 📊 HASIL KLASIFIKASI
# ===========================================
elif st.session_state.img:
    label, confidence, pred_probs, resized_img, entropy, gap, numeric_array = predict_image(
        st.session_state.img
    )

    img_bytes = convert_image_to_bytes(resized_img)
    img_base64 = base64.b64encode(img_bytes).decode()

    st.markdown("## Hasil Klasifikasi")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(resized_img, use_container_width=True)

    with col2:
        if label == "manusia":
            st.markdown("<div class='result-box'><p><strong>Status:</strong> Terdeteksi Wajah Manusia</p><p>Gambar ini bukan sampah. Silakan gunakan gambar lain untuk klasifikasi sampah anorganik.</p></div>", unsafe_allow_html=True)

        elif label == "unknown":
            st.markdown("<div class='result-box'><p><strong>Status:</strong> Tidak Terkelompokkan</p><p>Maaf, sistem tidak dapat mengenali gambar ini. Silakan unggah gambar lain.</p></div>", unsafe_allow_html=True)
            if st.button("🚨 Gunakan Meskipun Tidak Pasti"):
                st.session_state.force_classify = True
                st.rerun()
        else:
            st.session_state.force_classify = False
            st.markdown(f"""
            <div class='result-box'>
                <p><strong>📦 Kategori:</strong> Anorganik</p>
                <p><strong>🗑️ Jenis:</strong> {label.capitalize()}</p>
                <p><strong>📖 Penjelasan:</strong> {penjelasan[label]}</p>
                <p><strong>⏳ Waktu Daur Ulang:</strong> {waktu_daur_ulang[label]}</p>
                <a href="data:image/png;base64,{img_base64}" download="sampah_{label}.png">
                    <button style="background:#558b2f;color:white;padding:8px 20px;border:none;border-radius:10px;margin-top:12px;">
                        📅 Download Gambar Hasil
                    </button>
                </a>
            </div>
            """, unsafe_allow_html=True)

    if label not in ["unknown", "manusia"]:
        with st.expander("📊 Probabilitas Tiap Kelas"):
            for i, prob in enumerate(pred_probs):
                color_class = "high" if prob >= 0.8 else "medium" if prob >= 0.5 else "low"
                st.markdown(f"""
                <div class='prob-wrapper'>
                    <strong>{class_labels[i].capitalize()}</strong>:
                    <div class='prob-bar'>
                        <div class='prob-bar-fill {color_class}' style='width:{prob*100:.2f}%'>{prob*100:.2f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            df_bar = pd.DataFrame({"Kelas": class_labels, "Probabilitas": [round(p * 100, 2) for p in pred_probs]})
            st.bar_chart(df_bar.set_index("Kelas"))

        with st.expander("📷 Representasi Citra"):
            mean_rgb = np.mean(numeric_array, axis=(0, 1))
            df_rgb = pd.DataFrame({
                "Warna": ["R (Merah)", "G (Hijau)", "B (Biru)"],
                "Nilai Rata-Rata": [round(mean_rgb[0], 2), round(mean_rgb[1], 2), round(mean_rgb[2], 2)]
            })
            st.table(df_rgb)
            st.code(np.round(numeric_array[:5, :5, :], 2).__str__(), language="python")

    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔄 Re-upload Image"):
            st.session_state.img = None
            st.session_state.show_input_box = True
            st.session_state.force_classify = False
            st.session_state.show_edu = False
            st.rerun()

    with col4:
        if label not in ["unknown", "manusia"]:
            if st.button("📋 Education Information"):
                st.session_state.show_edu = True
                st.rerun()
