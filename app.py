import os
from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np

from fuzzy_system import sistem_fuzzy_beasiswa

app = Flask(__name__)
app.secret_key = 'kunci_rahasia_spk_beasiswa' 

# print("Memuat model CNN, harap tunggu...")
# if not os.path.exists(MODEL_SERTIFIKAT) or not os.path.exists(MODEL_RUMAH):
#     models_loaded_status = False
# else:
#     model_sertifikat, model_rumah, kelas_sertifikat, kelas_rumah = load_all_models(
#         MODEL_SERTIFIKAT, MODEL_RUMAH
#     )
#     models_loaded_status = True
#     print("Model CNN berhasil dimuat. Server Flask siap.")

print("="*50)
print("SERVER BERJALAN DALAM MODE 'FUZZY-ONLY' (TANPA CNN)")
print("Skor CNN akan diinput secara manual.")
print("="*50)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            ipk = float(request.form['ipk'])
            penghasilan_ortu = int(request.form['penghasilan_ortu'])
   
            skor_cnn_prestasi = float(request.form['skor_cnn_prestasi'])
            skor_cnn_finansial = float(request.form['skor_cnn_finansial'])

            if not (0 <= skor_cnn_prestasi <= 100 and 0 <= skor_cnn_finansial <= 100):
                flash("Kesalahan: Skor CNN Manual harus di antara 0 dan 100.")
                return redirect(request.url)

        except ValueError:
            flash("Kesalahan: Semua input harus berupa angka yang valid.")
            return redirect(request.url)

        skor_akhir = sistem_fuzzy_beasiswa(
            ipk_val=ipk,
            penghasilan_val=penghasilan_ortu,
            skor_cnn_prestasi_val=skor_cnn_prestasi,
            skor_cnn_finansial_val=skor_cnn_finansial
        )

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

        return render_template(
            'index.html',
            skor_akhir=f"{skor_akhir:.2f}",
            rekomendasi_teks=rekomendasi_teks,
            rekomendasi_level=rekomendasi_level,
            ipk_in=ipk,
            penghasilan_in=penghasilan_ortu,
            skor_cnn_prestasi_in=skor_cnn_prestasi, 
            skor_cnn_finansial_in=skor_cnn_finansial 
        )

    return render_template('index.html', skor_akhir=None)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)