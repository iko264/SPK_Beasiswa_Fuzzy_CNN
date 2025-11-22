import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


TARGET_SIZE = (150, 150) 

KELAS_SERTIFIKAT_MAP = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'} 
KELAS_RUMAH_MAP = {0: 'Mewah', 1: 'Sederhana', 2: 'Sedang'}

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


@tf.keras.utils.register_keras_serializable()
def load_all_models(model_path_sertifikat, model_path_rumah):
    """Memuat kedua model CNN dari file .h5."""
    print("Memuat model CNN...")
    try:
        model_sertifikat = load_model(model_path_sertifikat)
        print("Model Sertifikat OK.")
    except Exception as e:
        print(f"Gagal memuat model sertifikat: {e}")
        model_sertifikat = None

    try:
        model_rumah = load_model(model_path_rumah)
        print("Model Rumah OK.")
    except Exception as e:
        print(f"Gagal memuat model rumah: {e}")
        model_rumah = None
        
    print("Semua model dimuat.")
    return model_sertifikat, model_rumah, KELAS_SERTIFIKAT_MAP, KELAS_RUMAH_MAP


def get_cnn_fuzzy_score(image_input, model, class_map, score_logic):
    """
    Melakukan prediksi gambar dan mengkonversinya menjadi skor 0-100.
    'image_input' bisa berupa path file (str) atau BytesIO (dari Streamlit).
    """
    if model is None:
        print("Error: Model tidak tersedia. Mengembalikan skor 50.")
        return 50.0 

    try:
        img = image.load_img(image_input, target_size=TARGET_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0 
    except Exception as e:
        print(f"Error memproses gambar: {e}")
        return 50.0

    predictions = model.predict(img_array)[0]
    predicted_class_index = np.argmax(predictions)
    
    predicted_class_name = class_map.get(predicted_class_index, "Unknown")

    score_range = score_logic.get(predicted_class_name)
    
    if score_range:
        min_score, max_score = score_range
        fuzzy_score = (min_score + max_score) / 2
        print(f"Prediksi: {predicted_class_name} (Prob: {predictions[predicted_class_index]:.2f}) -> Skor: {fuzzy_score}")
        return fuzzy_score
    else:
        print(f"Error: Kelas '{predicted_class_name}' tidak ada dalam logika skor.")
        return 0.0