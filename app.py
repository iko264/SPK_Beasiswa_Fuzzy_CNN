import os
from flask import Flask, request, render_template, flash
from werkzeug.utils import secure_filename

from fuzzy_system import sistem_fuzzy_beasiswa
from cnn_utils import load_model_rumah, prediksi_rumah

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_RUMAH_PATH = "models/model_rumah.h5"

app = Flask(__name__)
app.secret_key = "kunci_rahasia_spk_beasiswa"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_rumah, kelas_rumah_map = load_model_rumah(MODEL_RUMAH_PATH)

SERTIFIKAT_TINGKAT_SCORE = {'': None, 'lokal': 60, 'nasional': 80, 'internasional': 95}
NOMINASI_SCORE = {'': None, 'peserta': 40, 'finalis': 60, 'nominasi': 75}
JUARA_SCORE = {'': None, 'tidak': 40, '3': 70, '2': 85, '1': 95}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        "ipk_in": 3.5,
        "penghasilan_in": 3000000,
        "sert_tingkat_in": "",
        "sert_nominasi_in": "",
        "sert_juara_in": "",
        "skor_cnn_prestasi_in": None,
        "skor_cnn_finansial_in": None,
        "skor_akhir": None,
        "rekomendasi_level": None,
        "rekomendasi_teks": None,
    }

    if request.method == 'POST':
        try:
            ipk = float(request.form.get('ipk', 3.5))
            penghasilan = float(request.form.get('penghasilan_ortu', 0))
        except ValueError:
            flash("IPK atau penghasilan harus berupa angka.")
            return render_template("index.html", **context)

        tingkat = request.form.get('sert_tingkat', '')
        nominasi = request.form.get('sert_nominasi', '')
        juara = request.form.get('sert_juara', '')

        vals = []
        if tingkat and SERTIFIKAT_TINGKAT_SCORE[tingkat] is not None:
            vals.append(SERTIFIKAT_TINGKAT_SCORE[tingkat])
        if nominasi and NOMINASI_SCORE[nominasi] is not None:
            vals.append(NOMINASI_SCORE[nominasi])
        if juara and JUARA_SCORE[juara] is not None:
            vals.append(JUARA_SCORE[juara])

        skor_prestasi = sum(vals) / len(vals) if vals else 50.0

        skor_finansial = 50.0
        if 'rumah_image' in request.files:
            file = request.files['rumah_image']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)

                if model_rumah is not None:
                    skor_finansial = prediksi_rumah(path, model_rumah, kelas_rumah_map)

        try:
            skor_akhir = sistem_fuzzy_beasiswa(
                ipk_val=ipk,
                penghasilan_val=penghasilan,
                prestasi_val=skor_prestasi,
                finansial_val=skor_finansial
            )
        except Exception as e:
            flash(f"Gagal menghitung fuzzy: {e}")
            return render_template("index.html", **context)

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
            "penghasilan_in": penghasilan,
            "sert_tingkat_in": tingkat,
            "sert_nominasi_in": nominasi,
            "sert_juara_in": juara,
            "skor_cnn_prestasi_in": skor_prestasi,
            "skor_cnn_finansial_in": skor_finansial,
            "skor_akhir": round(float(skor_akhir), 2),
            "rekomendasi_level": rekomendasi_level,
            "rekomendasi_teks": rekomendasi_teks,
        })

    return render_template("index.html", **context)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    