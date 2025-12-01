import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def load_model_rumah(path):
    if not os.path.exists(path):
        print("Model rumah tidak ditemukan.")
        return None, None

    try:
        model = load_model(path)
        kelas_map = {0: "kelas_mewah", 1: "kelas_sedang", 2: "kelas_sederhana"}
        print("Model rumah berhasil dimuat.")
        return model, kelas_map
    except Exception as e:
        print("Gagal load model:", e)
        return None, None


def prediksi_rumah(img_path, model, kelas_map):
    img = load_img(img_path, target_size=(128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    kelas_idx = np.argmax(pred)

    # skor finansial dibuat versi fuzzy:
    if kelas_map[kelas_idx] == "kelas_mewah":
        return 30
    elif kelas_map[kelas_idx] == "kelas_sedang":
        return 60
    else:
        return 85
