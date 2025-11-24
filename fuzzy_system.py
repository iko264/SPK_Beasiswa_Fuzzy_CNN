# fuzzy_system.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def sistem_fuzzy_beasiswa(ipk_val, penghasilan_val, skor_cnn_prestasi_val, skor_cnn_finansial_val):
    """
    Bangun & jalankan sistem fuzzy untuk rekomendasi beasiswa.
    Mengembalikan nilai antara 0 - 100.
    """

    # Validasi & konversi
    ipk_val = float(ipk_val)
    penghasilan_val = float(penghasilan_val)
    skor_cnn_prestasi_val = float(skor_cnn_prestasi_val)
    skor_cnn_finansial_val = float(skor_cnn_finansial_val)

    # Universe of discourse
    ipk_ud = np.arange(0, 4.01, 0.01)
    penghasilan_ud = np.arange(0, 15000001, 50000)
    cnn_score_ud = np.arange(0, 101, 1)
    prioritas_ud = np.arange(0, 101, 1)

    # Antecedents & Consequent
    ipk = ctrl.Antecedent(ipk_ud, 'ipk')
    penghasilan = ctrl.Antecedent(penghasilan_ud, 'penghasilan')
    skor_cnn_prestasi = ctrl.Antecedent(cnn_score_ud, 'skor_cnn_prestasi')
    skor_cnn_finansial = ctrl.Antecedent(cnn_score_ud, 'skor_cnn_finansial')
    prioritas = ctrl.Consequent(prioritas_ud, 'prioritas', defuzzify_method='centroid')

    # Membership functions
    ipk['rendah'] = fuzz.trimf(ipk_ud, [0.0, 2.0, 3.0])
    ipk['sedang'] = fuzz.trimf(ipk_ud, [2.5, 3.2, 3.6])
    ipk['tinggi'] = fuzz.trimf(ipk_ud, [3.3, 4.0, 4.0])

    penghasilan['rendah'] = fuzz.trimf(penghasilan_ud, [0, 1000000, 4000000])
    penghasilan['sedang'] = fuzz.trimf(penghasilan_ud, [3000000, 6000000, 9000000])
    penghasilan['tinggi'] = fuzz.trimf(penghasilan_ud, [7000000, 15000000, 15000000])

    skor_cnn_prestasi['kurang'] = fuzz.trimf(cnn_score_ud, [0, 0, 60])
    skor_cnn_prestasi['baik'] = fuzz.trimf(cnn_score_ud, [50, 70, 85])
    skor_cnn_prestasi['sangat_baik'] = fuzz.trimf(cnn_score_ud, [75, 100, 100])

    skor_cnn_finansial['buruk'] = fuzz.trimf(cnn_score_ud, [0, 0, 60])
    skor_cnn_finansial['sedang'] = fuzz.trimf(cnn_score_ud, [50, 70, 85])
    skor_cnn_finansial['baik'] = fuzz.trimf(cnn_score_ud, [75, 100, 100])

    prioritas['rendah'] = fuzz.trimf(prioritas_ud, [0, 10, 40])
    prioritas['sedang'] = fuzz.trimf(prioritas_ud, [30, 50, 70])
    prioritas['tinggi'] = fuzz.trimf(prioritas_ud, [60, 80, 90])
    prioritas['sangat_tinggi'] = fuzz.trimf(prioritas_ud, [85, 95, 100])

    # Rules 
    rules = [
        ctrl.Rule(ipk['tinggi'] & penghasilan['rendah'] & skor_cnn_prestasi['sangat_baik'] & skor_cnn_finansial['buruk'],
                  prioritas['sangat_tinggi']),
        ctrl.Rule(ipk['sedang'] & penghasilan['rendah'] & skor_cnn_prestasi['baik'] & skor_cnn_finansial['buruk'],
                  prioritas['tinggi']),
        ctrl.Rule(ipk['tinggi'] & penghasilan['sedang'] & skor_cnn_prestasi['sangat_baik'] & skor_cnn_finansial['sedang'],
                  prioritas['tinggi']),
        ctrl.Rule(penghasilan['tinggi'] | skor_cnn_finansial['baik'], prioritas['rendah']),
        ctrl.Rule(ipk['sedang'] & penghasilan['sedang'] & skor_cnn_prestasi['baik'] & skor_cnn_finansial['sedang'],
                  prioritas['sedang']),
        ctrl.Rule(ipk['rendah'] & penghasilan['rendah'] & skor_cnn_prestasi['kurang'] & skor_cnn_finansial['buruk'],
                  prioritas['sedang'])
    ]

    control = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(control)

    # Input ke sistem
    simulation.input['ipk'] = ipk_val
    simulation.input['penghasilan'] = penghasilan_val
    simulation.input['skor_cnn_prestasi'] = skor_cnn_prestasi_val
    simulation.input['skor_cnn_finansial'] = skor_cnn_finansial_val

    simulation.compute()

    return float(simulation.output['prioritas'])

