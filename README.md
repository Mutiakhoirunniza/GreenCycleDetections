# 💚 GreenCycleDetections

**GreenCycleDetections** adalah aplikasi berbasis web untuk mengklasifikasikan jenis sampah anorganik menggunakan citra digital. Aplikasi ini dibangun sebagai bagian dari tugas akhir/skripsi untuk memberikan solusi praktis dalam mendeteksi dan mengedukasi masyarakat mengenai daur ulang sampah anorganik.

---

## 🧪 Latar Belakang

Sampah anorganik seperti plastik, kaca, logam, dan kertas merupakan jenis limbah yang sulit terurai secara alami. Pengelolaan yang kurang tepat dapat menyebabkan pencemaran lingkungan jangka panjang. Oleh karena itu, dibutuhkan sistem yang dapat mengklasifikasikan sampah anorganik secara otomatis agar proses daur ulang menjadi lebih efisien dan edukatif.

---

## 🎯 Tujuan Proyek

- Mengembangkan model klasifikasi citra sampah anorganik berbasis CNN menggunakan **EfficientNetB0**.
- Membuat aplikasi Streamlit interaktif untuk klasifikasi gambar sampah.
- Menyediakan fitur edukasi seputar jenis-jenis sampah dan pentingnya daur ulang.

---

## 📦 Fitur Utama

- 🔍 Klasifikasi Citra Sampah (Plastik, Kaca, Logam, Kertas)
- 🧠 Model **EfficientNetB0** yang telah dilatih
- 📊 Probabilitas Prediksi per Kelas & Confidence Score
- 🕒 Waktu Klasifikasi per Gambar
- 📥 Download Gambar dengan Label
- 📚 Halaman Edukasi Interaktif

---

## 🛠️ Teknologi yang Digunakan

- Python 3.10
- TensorFlow & Keras
- EfficientNetB0 (Transfer Learning)
- Streamlit (Web App)
- OpenCV, NumPy, Pandas
- Matplotlib & Seaborn
- PIL (Image Processing)

---

## 🚀 Cara Menjalankan Aplikasi

### Opsi 1: Vercel (Rekomendasi)
Aplikasi ini sekarang mendukung deployment ke **Vercel** sebagai Web App modern.
1. Hubungkan repositori GitHub Anda ke Vercel.
2. Vercel akan secara otomatis mendeteksi `api/index.py` sebagai Serverless Function.
3. Klik **Deploy**.

### Opsi 2: Lokal (FastAPI + Modern UI)
1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan backend:
   ```bash
   uvicorn api.index:app --reload
   ```
3. Buka `index.html` di browser Anda.

### Opsi 3: Streamlit (Legacy)
Jika ingin menggunakan versi lama:
```bash
streamlit run klasifikasi.py
```


---
## 🔗 Notebook Google Colab

Klik tombol di bawah untuk membuka notebook training model di Google Colab:

[![Buka di Google Colab](https://colab.research.google.com/drive/1xXzMAZZs5B7mA-GwucOfntKU0K8vOFi-?usp=sharing)

---

## 👨‍🎓 Informasi Skripsi

**Judul Skripsi**:  
*“Klasifikasi Jenis Sampah Anorganik Menggunakan Algoritma CNN berbasis Citra Digital”*

**Nama**: Diah Mutia Choirunnisa
**NIM**: 2170231001  
**Program Studi**: Teknik Informatika  
**Universitas**: Universitas Krisnadwipayana  
**Tahun**: 2021

---

## 📃 Lisensi

Lisensi proyek ini menggunakan [MIT License](LICENSE). Bebas digunakan dan dimodifikasi dengan mencantumkan atribusi.

---

## 🙏 Ucapan Terima Kasih

Terima kasih kepada dosen pembimbing, rekan-rekan seperjuangan, dan semua pihak yang telah memberikan dukungan moral maupun teknis selama proses penelitian dan pengembangan aplikasi ini.

---

