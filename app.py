import os
from flask import Flask, request, render_template, redirect, flash, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from fuzzy_system import sistem_fuzzy_beasiswa  

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

MODEL_RUMAH_PATH = "models/model_rumah.h5"  

app = Flask(__name__)
app.secret_key = "kunci_rahasia_spk_beasiswa"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# load model cnn rumah
model_rumah = None
kelas_rumah_map = ['kelas_mewah', 'kelas_sedang', 'kelas_sederhana']  

if os.path.exists(MODEL_RUMAH_PATH):
    try:
        model_rumah = load_model(MODEL_RUMAH_PATH)
        print("Model rumah ditemukan dan dimuat.")
    except Exception as e:
        print(f"Gagal memuat model rumah: {e}")
else:
    print("Model rumah tidak ditemukan — aplikasi akan menggunakan input finansial manual.")

# manual scoring mapping
SERTIFIKAT_TINGKAT_SCORE = {
    '': None,
    'lokal': 60,
    'nasional': 80,
    'internasional': 95
}
NOMINASI_SCORE = {
    '': None,
    'peserta': 40,
    'finalis': 60,
    'nominasi': 75
}
JUARA_SCORE = {
    '': None,
    'tidak': 40,
    '3': 70,
    '2': 85,
    '1': 95
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#prediksi rumah dan kembalikan skor finansial fuzzy
def prediksi_rumah(path):
    img = load_img(path, target_size=(128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model_rumah.predict(img)
    idx = np.argmax(pred)

    kelas = kelas_rumah_map[idx]

    # mapping fuzzy nilai finansial berdasarkan kelas rumah
    if kelas == 'kelas_mewah':
        return 20     
    elif kelas == 'kelas_sedang':
        return 50
    elif kelas == 'kelas_sederhana':
        return 80     

    return 50

#route utama
@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        "ipk_in": 3.5,
        "penghasilan_in": 3000000,
        "skor_cnn_prestasi_in": 80,
        "skor_cnn_finansial_in": 50,
        "sert_tingkat_in": "",
        "sert_nominasi_in": "",
        "sert_juara_in": "",
        "skor_akhir": None,
        "rekomendasi_level": None,
        "rekomendasi_teks": None,
    }

    if request.method == 'POST':

        # input IPK dan penghasilan
        try:
            ipk = float(request.form.get('ipk', context["ipk_in"]))
            penghasilan_ortu = float(request.form.get('penghasilan_ortu', context["penghasilan_in"]))
        except ValueError:
            flash("Input IPK atau penghasilan tidak valid.")
            return render_template("index.html", **context)

        # hitung skor prestasi (manual)
        tingkat = request.form.get('sert_tingkat', '')
        nominasi = request.form.get('sert_nominasi', '')
        juara = request.form.get('sert_juara', '')

        skor_prestasi_manual = request.form.get('skor_cnn_prestasi', '').strip()

        if skor_prestasi_manual != '':
            try:
                skor_prestasi = float(skor_prestasi_manual)
            except:
                flash("Skor prestasi harus angka.")
                return render_template("index.html", **context)
        else:
            vals = []
            if tingkat != '' and SERTIFIKAT_TINGKAT_SCORE[tingkat] is not None:
                vals.append(SERTIFIKAT_TINGKAT_SCORE[tingkat])
            if nominasi != '' and NOMINASI_SCORE[nominasi] is not None:
                vals.append(NOMINASI_SCORE[nominasi])
            if juara != '' and JUARA_SCORE[juara] is not None:
                vals.append(JUARA_SCORE[juara])

            skor_prestasi = sum(vals) / len(vals) if vals else 50.0


        # hitung skor finansial (manual)
        skor_finansial_manual = request.form.get('skor_cnn_finansial', '').strip()
        skor_finansial = None

        if 'rumah_image' in request.files:
            file = request.files['rumah_image']
            if file and file.filename != '' and allowed_file(file.filename):

                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)

                if model_rumah is not None:
                    skor_finansial = prediksi_rumah(save_path)
                else:
                    skor_finansial = None

        # Jika CNN tidak menghasilkan nilai → pakai input manual
        if skor_finansial is None:
            if skor_finansial_manual != '':
                try:
                    skor_finansial = float(skor_finansial_manual)
                except:
                    flash("Skor finansial harus angka.")
                    return render_template("index.html", **context)
            else:
                skor_finansial = 50.0

        # cek validitas skor
        if not (0 <= skor_prestasi <= 100):
            flash("Skor prestasi harus 0–100.")
            return render_template("index.html", **context)

        if not (0 <= skor_finansial <= 100):
            flash("Skor finansial harus 0–100.")
            return render_template("index.html", **context)


        # hitung fuzzy
        skor_akhir = sistem_fuzzy_beasiswa(
            ipk_val=ipk,
            penghasilan_val=penghasilan_ortu,
            skor_cnn_prestasi_val=skor_prestasi,
            skor_cnn_finansial_val=skor_finansial
        )


        # output rekomendasi
        rekomendasi_teks = "Prioritas RENDAH"
        rekomendasi_level = "rendah"

        if skor_akhir >= 85:
            rekomendasi_teks = "Prioritas SANGAT TINGGI"
            rekomendasi_level = "sangat-tinggi"
        elif skor_akhir >= 65:
            rekomendasi_teks = "Prioritas TINGGI"
            rekomendasi_level = "tinggi"
        elif skor_akhir >= 40:
            rekomendasi_teks = "Prioritas SEDANG"
            rekomendasi_level = "sedang"

        context.update({
            "ipk_in": ipk,
            "penghasilan_in": penghasilan_ortu,
            "skor_cnn_prestasi_in": skor_prestasi,
            "skor_cnn_finansial_in": skor_finansial,
            "sert_tingkat_in": tingkat,
            "sert_nominasi_in": nominasi,
            "sert_juara_in": juara,
            "skor_akhir": round(float(skor_akhir), 2),
            "rekomendasi_level": rekomendasi_level,
            "rekomendasi_teks": rekomendasi_teks
        })

        return render_template("index.html", **context)

    return render_template("index.html", **context)



if __name__ == '__main__':
    print("="*60)
    print("SERVER STARTED - FUZZY + CNN(RUMAH)")
    print("http://127.0.0.1:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
