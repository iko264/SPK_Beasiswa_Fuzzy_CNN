# app.py
import os
from flask import Flask, request, render_template, redirect, flash, url_for
from werkzeug.utils import secure_filename
import numpy as np

from fuzzy_system import sistem_fuzzy_beasiswa  # fungsi fuzzy utama
from cnn_utils import load_all_models, get_cnn_fuzzy_score

# Konfigurasi
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_RUMAH_PATH = "models/model_cnn_rumah.h5"  # letakkan model rumah di sini jika ada

app = Flask(__name__)
app.secret_key = "kunci_rahasia_spk_beasiswa"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Coba muat model rumah (opsional)
model_rumah = None
kelas_rumah_map = None
if os.path.exists(MODEL_RUMAH_PATH):
    try:
        model_rumah, _, _, kelas_rumah_map = load_all_models(None, MODEL_RUMAH_PATH)
        print("Model rumah ditemukan dan dimuat.")
    except Exception as e:
        print(f"Gagal memuat model rumah: {e}")
else:
    print("Model rumah tidak ditemukan — aplikasi berjalan tanpa CNN rumah (akan pakai skor default/manual).")

# Mapping dropdown sertifikat -> skor (simple mapping)
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

@app.route('/', methods=['GET', 'POST'])
def index():
    # default values to avoid Jinja undefined errors
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
        # Baca input numerik dasar
        try:
            ipk = float(request.form.get('ipk', context["ipk_in"]))
            penghasilan_ortu = float(request.form.get('penghasilan_ortu', context["penghasilan_in"]))
        except ValueError:
            flash("Input IPK atau penghasilan tidak valid. Gunakan angka.")
            return render_template("index.html", **context)

        # Ambil informasi sertifikat (dropdown) — tetap manual
        tingkat = request.form.get('sert_tingkat', '')
        nominasi = request.form.get('sert_nominasi', '')
        juara = request.form.get('sert_juara', '')

        # Pilihan: user dapat memasukkan skor prestasi manual (mengoverride dropdown)
        skor_prestasi_manual = request.form.get('skor_cnn_prestasi', '').strip()
        if skor_prestasi_manual != '':
            try:
                skor_prestasi = float(skor_prestasi_manual)
            except ValueError:
                flash("Skor prestasi harus angka antara 0-100.")
                return render_template("index.html", **context)
        else:
            # jika tidak ada input manual, gabungkan dropdown menjadi skor rata-rata dari mapping yang ada
            vals = []
            if tingkat in SERTIFIKAT_TINGKAT_SCORE and SERTIFIKAT_TINGKAT_SCORE[tingkat] is not None:
                vals.append(SERTIFIKAT_TINGKAT_SCORE[tingkat])
            if nominasi in NOMINASI_SCORE and NOMINASI_SCORE[nominasi] is not None:
                vals.append(NOMINASI_SCORE[nominasi])
            if juara in JUARA_SCORE and JUARA_SCORE[juara] is not None:
                vals.append(JUARA_SCORE[juara])
            skor_prestasi = float(sum(vals) / len(vals)) if vals else 50.0  # default 50 jika tidak diisi

        # Skor finansial: dari CNN rumah (upload) atau manual angka
        skor_finansial_manual = request.form.get('skor_cnn_finansial', '').strip()
        skor_finansial = None

        # Cek apakah ada file rumah diupload
        if 'rumah_image' in request.files:
            file = request.files['rumah_image']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                # Jika model tersedia, gunakan CNN untuk skor rumah. Jika tidak, fallback ke manual/default.
                if model_rumah is not None and kelas_rumah_map is not None:
                    try:
                        skor_finansial = get_cnn_fuzzy_score(save_path, model_rumah, kelas_rumah_map, None)
                        # get_cnn_fuzzy_score pada implementasimu mengharapkan score_logic; jika None, fungsi harus handle dan kembalikan nilai (50)
                        # Kita asumsikan get_cnn_fuzzy_score mengembalikan float 0-100.
                    except Exception as e:
                        print(f"Error prediksi CNN rumah: {e}")
                        skor_finansial = None
                else:
                    # model rumah tidak ada -> pakai default / manual jika ada
                    skor_finansial = None

        # jika tidak ada file atau CNN gagal, gunakan input manual jika ada
        if skor_finansial is None:
            if skor_finansial_manual != '':
                try:
                    skor_finansial = float(skor_finansial_manual)
                except ValueError:
                    flash("Skor finansial harus angka antara 0-100.")
                    return render_template("index.html", **context)
            else:
                skor_finansial = 50.0  # default

        # Validasi rentang skor 0-100
        for s_val, name in [(skor_prestasi, "Prestasi"), (skor_finansial, "Finansial")]:
            if not (0 <= float(s_val) <= 100):
                flash(f"Nilai {name} harus antara 0 dan 100.")
                return render_template("index.html", **context)

        # Panggil sistem fuzzy utama
        try:
            skor_akhir = sistem_fuzzy_beasiswa(
                ipk_val=ipk,
                penghasilan_val=penghasilan_ortu,
                skor_cnn_prestasi_val=skor_prestasi,
                skor_cnn_finansial_val=skor_finansial
            )
        except Exception as e:
            flash(f"Error pada sistem fuzzy: {e}")
            return render_template("index.html", **context)

        # Interpretasi rekomendasi
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

        # Update context untuk render
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

    # GET -> render dengan defaults
    return render_template("index.html", **context)


if __name__ == '__main__':
    print("="*60)
    print("SERVER STARTED - FUZZY + CNN(RUMAH) MODE (CNN RUMAH OPSIONAL)")
    print("Open http://127.0.0.1:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
