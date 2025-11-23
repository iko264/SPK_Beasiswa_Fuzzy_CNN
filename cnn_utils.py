# cnn_utils.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

TARGET_SIZE = (150, 150)

# Map kelas model rumah — sesuaikan dengan mapping ketika kamu melatih model
# Pastikan ini sesuai dengan class_indices dari ImageDataGenerator saat training
KELAS_RUMAH_MAP = {0: 'Sederhana', 1: 'Sedang', 2: 'Mewah'}

# Skor logic default (jika ingin mengkonversi class->range)
SKOR_LOGIC_RUMAH = {
    'Sederhana': (0, 55),
    'Sedang': (56, 80),
    'Mewah': (81, 100)
}

def load_all_models(model_path_sertifikat=None, model_path_rumah=None):
    """
    Memuat model sertifikat (opsional) dan model rumah (opsional).
    Mengembalikan (model_sertifikat, model_rumah, kelas_sertifikat_map, kelas_rumah_map)
    Jika suatu model tidak ada -> None.
    """
    model_sertifikat = None
    model_rumah = None

    if model_path_sertifikat and os.path.exists(model_path_sertifikat):
        try:
            model_sertifikat = load_model(model_path_sertifikat)
            print("Model sertifikat dimuat.")
        except Exception as e:
            print(f"Gagal memuat model sertifikat: {e}")

    if model_path_rumah and os.path.exists(model_path_rumah):
        try:
            model_rumah = load_model(model_path_rumah)
            print("Model rumah dimuat.")
        except Exception as e:
            print(f"Gagal memuat model rumah: {e}")

    return model_sertifikat, model_rumah, None, KELAS_RUMAH_MAP

def get_cnn_fuzzy_score(image_input_path, model, class_map, score_logic=None):
    """
    Prediksi gambar menggunakan model dan kembalikan skor (0-100).
    - image_input_path: path file gambar
    - model: model keras (atau None)
    - class_map: mapping index->label
    - score_logic: dict label->(min,max) atau None
    """
    if model is None:
        print("Model CNN tidak tersedia, kembalikan skor default 50.0")
        return 50.0

    try:
        img = image.load_img(image_input_path, target_size=TARGET_SIZE)
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0) / 255.0
    except Exception as e:
        print(f"Error memuat gambar: {e}")
        return 50.0

    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    label = class_map.get(idx, "Unknown")

    if score_logic and label in score_logic:
        mn, mx = score_logic[label]
        return float((mn + mx) / 2.0)
    else:
        # jika score_logic None, gunakan probabilitas untuk skala 0-100
        prob = float(preds[idx])
        return float(round(prob * 100.0, 2))
