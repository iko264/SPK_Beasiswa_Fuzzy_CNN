import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

TARGET_SIZE = (150, 150)

KELAS_SERTIFIKAT_MAP = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
KELAS_RUMAH_MAP = {0: 'Sederhana', 1: 'Sedang', 2: 'Mewah'}

SKOR_LOGIC_SERTIFIKAT = {
    'Rendah': (0, 55),
    'Sedang': (56, 80),
    'Tinggi': (81, 100)
}

SKOR_LOGIC_RUMAH = {
    'Sederhana': (0, 55),
    'Sedang': (56, 80),
    'Mewah': (81, 100)
}

def load_all_models(model_path_sertifikat, model_path_rumah):
    """Memuat kedua model CNN dari file .h5."""
    print("Memuat model CNN...")
    model_sertifikat = None
    model_rumah = None
    try:
        if model_path_sertifikat and os.path.exists(model_path_sertifikat):
            model_sertifikat = load_model(model_path_sertifikat)
            print("Model Sertifikat OK.")
        else:
            print("Model sertifikat tidak ditemukan atau path kosong.")
    except Exception as e:
        print(f"Gagal memuat model sertifikat: {e}")

    try:
        if model_path_rumah and os.path.exists(model_path_rumah):
            model_rumah = load_model(model_path_rumah)
            print("Model Rumah OK.")
        else:
            print("Model rumah tidak ditemukan atau path kosong.")
    except Exception as e:
        print(f"Gagal memuat model rumah: {e}")

    print("Selesai memuat model.")
    return model_sertifikat, model_rumah, KELAS_SERTIFIKAT_MAP, KELAS_RUMAH_MAP

def get_cnn_fuzzy_score(image_input, model, class_map, score_logic):
    """
    Melakukan prediksi gambar dan mengkonversinya menjadi skor 0-100.
    'image_input' bisa berupa path file (str) atau file-like object.
    """
    if model is None:
        print("Warning: Model tidak tersedia. Mengembalikan skor default 50.0")
        return 50.0

    try:
        img = image.load_img(image_input, target_size=TARGET_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
    except Exception as e:
        print(f"Error memproses gambar: {e}")
        return 50.0

    try:
        predictions = model.predict(img_array)
        if predictions is None or len(predictions) == 0:
            return 50.0
        preds = predictions[0]
        predicted_class_index = int(np.argmax(preds))
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        return 50.0

    predicted_class_name = class_map.get(predicted_class_index, "Unknown")
    score_range = score_logic.get(predicted_class_name)

    if score_range:
        min_score, max_score = score_range
        fuzzy_score = (min_score + max_score) / 2.0
        print(f"Prediksi: {predicted_class_name} (Prob: {preds[predicted_class_index]:.3f}) -> Skor: {fuzzy_score}")
        return float(fuzzy_score)
    else:
        print(f"Error: Kelas '{predicted_class_name}' tidak ada dalam logika skor.")
        return 0.0
