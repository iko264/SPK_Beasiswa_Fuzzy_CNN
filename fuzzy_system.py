import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def sistem_fuzzy_beasiswa(ipk_val, penghasilan_val, prestasi_val, finansial_val):

    ipk_val = float(ipk_val)
    penghasilan_val = float(penghasilan_val)
    prestasi_val = float(prestasi_val)
    finansial_val = float(finansial_val)

    # Domain
    ipk_ud = np.arange(0, 4.01, 0.01)
    penghasilan_ud = np.arange(0, 15000001, 50000)
    skor_ud = np.arange(0, 101, 1)
    prioritas_ud = np.arange(0, 101, 1)

    # Variabel fuzzy
    ipk = ctrl.Antecedent(ipk_ud, 'ipk')
    penghasilan = ctrl.Antecedent(penghasilan_ud, 'penghasilan')
    prestasi = ctrl.Antecedent(skor_ud, 'prestasi')
    finansial = ctrl.Antecedent(skor_ud, 'finansial')
    prioritas = ctrl.Consequent(prioritas_ud, 'prioritas', defuzzify_method='centroid')

    # Membership functions
    ipk['rendah'] = fuzz.trimf(ipk_ud, [0.0, 2.0, 3.0])
    ipk['sedang'] = fuzz.trimf(ipk_ud, [2.5, 3.2, 3.6])
    ipk['tinggi'] = fuzz.trimf(ipk_ud, [3.3, 4.0, 4.0])

    penghasilan['rendah'] = fuzz.trimf(penghasilan_ud, [0, 1500000, 4000000])
    penghasilan['sedang'] = fuzz.trimf(penghasilan_ud, [3000000, 6000000, 9000000])
    penghasilan['tinggi'] = fuzz.trimf(penghasilan_ud, [7000000, 15000000, 15000000])

    prestasi['kurang'] = fuzz.trimf(skor_ud, [0, 0, 60])
    prestasi['baik'] = fuzz.trimf(skor_ud, [50, 70, 85])
    prestasi['sangat_baik'] = fuzz.trimf(skor_ud, [75, 100, 100])

    finansial['baik'] = fuzz.trimf(skor_ud, [0, 0, 60])     
    finansial['sedang'] = fuzz.trimf(skor_ud, [50, 70, 85])
    finansial['buruk'] = fuzz.trimf(skor_ud, [75, 100, 100])  

    # Output prioritas
    prioritas['rendah'] = fuzz.trimf(prioritas_ud, [0, 10, 40])
    prioritas['sedang'] = fuzz.trimf(prioritas_ud, [30, 50, 70])
    prioritas['tinggi'] = fuzz.trimf(prioritas_ud, [60, 80, 90])
    prioritas['sangat_tinggi'] = fuzz.trimf(prioritas_ud, [85, 95, 100])

    # RULES 
    rules = [
        ctrl.Rule(ipk['tinggi'] & penghasilan['rendah']
                  & prestasi['sangat_baik'] & finansial['buruk'],
                  prioritas['sangat_tinggi']),

        ctrl.Rule(ipk['sedang'] & penghasilan['rendah']
                  & prestasi['baik'] & finansial['buruk'],
                  prioritas['tinggi']),

        ctrl.Rule(penghasilan['rendah'] & finansial['buruk'],
                  prioritas['tinggi']),

        ctrl.Rule(penghasilan['tinggi'] | finansial['baik'],
                  prioritas['rendah']),

        ctrl.Rule(ipk['sedang'] & penghasilan['sedang']
                  & prestasi['baik'],
                  prioritas['sedang']),

        ctrl.Rule(ipk['rendah'] & prestasi['kurang'] & finansial['baik'],
                  prioritas['rendah'])
    ]

    control = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(control)

    # Input
    simulation.input['ipk'] = ipk_val
    simulation.input['penghasilan'] = penghasilan_val
    simulation.input['prestasi'] = prestasi_val
    simulation.input['finansial'] = finansial_val

    # Jalankan fuzzy (aman)
    try:
        simulation.compute()
    except:
        return 50.0

    if 'prioritas' not in simulation.output:
        return 50.0

    return float(simulation.output['prioritas'])
