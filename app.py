from flask import Flask, render_template, request, flash
import numpy as np

app = Flask(__name__)
app.secret_key = "beasiswa-secret-key"


def fuzzy_ipk(ipk):
    rendah = max(0, min((2.5 - ipk) / 1.0, 1))
    sedang = max(0, min((ipk - 2.0) / 0.5, (3.5 - ipk) / 0.5))
    tinggi = max(0, min((ipk - 3.0) / 1.0, 1))
    return rendah, sedang, tinggi


def fuzzy_penghasilan(p):
    rendah = max(0, min((6000000 - p) / 6000000, 1))
    sedang = max(0, min((p - 2000000) / 2000000, (8000000 - p) / 2000000))
    tinggi = max(0, min((p - 5000000) / 5000000, 1))
    return rendah, sedang, tinggi


def fuzzy_skor_manual(x):
    rendah = max(0, min((50 - x) / 50, 1))
    sedang = max(0, min((x - 30) / 20, (70 - x) / 20))
    tinggi = max(0, min((x - 60) / 40, 1))
    return rendah, sedang, tinggi

def fuzzy_inference(ipk, p, prestasi, finansial):
    ipk_r, ipk_s, ipk_t = fuzzy_ipk(ipk)
    p_r, p_s, p_t = fuzzy_penghasilan(p)
    pr_r, pr_s, pr_t = fuzzy_skor_manual(prestasi)
    fi_r, fi_s, fi_t = fuzzy_skor_manual(finansial)

    rule_sangat_tinggi = min(ipk_t, p_r, pr_t, fi_t)
    rule_tinggi = max(min(ipk_t, p_s), min(ipk_s, p_r), min(pr_t, fi_s))
    rule_sedang = max(min(ipk_s, p_s), min(pr_s, fi_s))
    rule_rendah = max(ipk_r, p_t, pr_r, fi_r)

    return {
        "sangat_tinggi": rule_sangat_tinggi,
        "tinggi": rule_tinggi,
        "sedang": rule_sedang,
        "rendah": rule_rendah
    }

def defuzzy(fz):
    values = {
        "sangat_tinggi": 90,
        "tinggi": 75,
        "sedang": 50,
        "rendah": 20
    }
    numerator = sum(fz[key] * values[key] for key in fz)
    denominator = sum(fz.values())
    if denominator == 0:
        return 0
    return round(numerator / denominator, 2)

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        try:
            ipk = float(request.form["ipk"])
            penghasilan = float(request.form["penghasilan_ortu"])
            prestasi = int(request.form["skor_cnn_prestasi"])
            finansial = int(request.form["skor_cnn_finansial"])

        except:
            flash("Input tidak valid. Pastikan semua angka terisi dengan benar.")
            return render_template("index.html")

        fz = fuzzy_inference(ipk, penghasilan, prestasi, finansial)
        skor = defuzzy(fz)

        if skor >= 85:
            level = "sangat-tinggi"
            teks = "Sangat Layak Mendapatkan Beasiswa"
        elif skor >= 70:
            level = "tinggi"
            teks = "Layak Mendapatkan Beasiswa"
        elif skor >= 50:
            level = "sedang"
            teks = "Dipertimbangkan (Sedang)"
        else:
            level = "rendah"
            teks = "Tidak Direkomendasikan"

        return render_template(
            "index.html",
            skor_akhir=skor,
            rekomendasi_level=level,
            rekomendasi_teks=teks,

            ipk_in=ipk,
            penghasilan_in=penghasilan,
            skor_cnn_prestasi_in=prestasi,
            skor_cnn_finansial_in=finansial
        )
        
    return render_template(
        "index.html",
        penghasilan_in = None,
        tanggungan_in = None,
        ipk_in = None
    )

if __name__ == "__main__":
    print("====================================================")
    print(" FLASK BERJALAN DI http://127.0.0.1:5000 ")
    print(" MODE: FUZZY ONLY (TANPA CNN) ")
    print("====================================================")
    app.run(debug=True)
