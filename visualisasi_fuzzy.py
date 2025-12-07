import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import os

# Buat folder untuk menyimpan gambar jika belum ada
OUTPUT_DIR = "gambar_laporan"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Sedang mengonfigurasi variabel fuzzy...")


ipk_ud = np.arange(0, 4.01, 0.01)
penghasilan_ud = np.arange(0, 15000001, 50000)
skor_ud = np.arange(0, 101, 1)
semester_ud = np.arange(1, 9, 1)
keaktifan_ud = np.arange(0, 101, 1)
prioritas_ud = np.arange(0, 101, 1)

ipk = ctrl.Antecedent(ipk_ud, 'IPK')
penghasilan = ctrl.Antecedent(penghasilan_ud, 'Penghasilan_Ortu')
prestasi = ctrl.Antecedent(skor_ud, 'Prestasi')
finansial = ctrl.Antecedent(skor_ud, 'Kondisi_Rumah_CNN')
semester = ctrl.Antecedent(semester_ud, 'Semester')
keaktifan = ctrl.Antecedent(keaktifan_ud, 'Keaktifan_Organisasi')
prioritas = ctrl.Consequent(prioritas_ud, 'Prioritas_Beasiswa')


ipk['rendah'] = fuzz.trimf(ipk_ud, [0.0, 2.0, 3.0])
ipk['sedang'] = fuzz.trimf(ipk_ud, [2.5, 3.2, 3.6])
ipk['tinggi'] = fuzz.trimf(ipk_ud, [3.3, 4.0, 4.0])

penghasilan['rendah'] = fuzz.trimf(penghasilan_ud, [0, 1500000, 4000000])
penghasilan['sedang'] = fuzz.trimf(penghasilan_ud, [3000000, 6000000, 9000000])
penghasilan['tinggi'] = fuzz.trimf(penghasilan_ud, [7000000, 15000000, 15000000])

for var in [prestasi, finansial]:
    var['kurang/buruk'] = fuzz.trimf(skor_ud, [0, 0, 60])
    var['cukup/sedang'] = fuzz.trimf(skor_ud, [50, 70, 85])
    var['baik/bagus'] = fuzz.trimf(skor_ud, [75, 100, 100])

semester['awal'] = fuzz.trimf(semester_ud, [1, 1, 3])
semester['tengah'] = fuzz.trimf(semester_ud, [2, 4, 6])
semester['akhir'] = fuzz.trimf(semester_ud, [5, 8, 8])

keaktifan['rendah'] = fuzz.trimf(keaktifan_ud, [0, 0, 40])
keaktifan['sedang'] = fuzz.trimf(keaktifan_ud, [30, 50, 70])
keaktifan['tinggi'] = fuzz.trimf(keaktifan_ud, [60, 100, 100])

prioritas['rendah'] = fuzz.trimf(prioritas_ud, [0, 10, 40])
prioritas['sedang'] = fuzz.trimf(prioritas_ud, [30, 50, 70])
prioritas['tinggi'] = fuzz.trimf(prioritas_ud, [60, 80, 90])
prioritas['sangat_tinggi'] = fuzz.trimf(prioritas_ud, [85, 95, 100])


variables = [ipk, penghasilan, prestasi, finansial, semester, keaktifan, prioritas]
filenames = ["1_ipk", "2_penghasilan", "3_prestasi", "4_rumah_cnn", "5_semester", "6_keaktifan", "7_output_prioritas"]

print("Mulai menyimpan gambar fungsi keanggotaan...")

for var, fname in zip(variables, filenames):
    try:
        var.view() 
        fig = plt.gcf() 
        fig.set_size_inches(8, 4) 
        save_path = os.path.join(OUTPUT_DIR, f"{fname}.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig) 
        print(f"  [OK] Tersimpan: {save_path}")
    except Exception as e:
        print(f"  [ERROR] Gagal menyimpan {fname}: {e}")

print("\nMenghitung Surface Plot 3D (IPK vs Penghasilan)...")

rules = [
    ctrl.Rule(ipk['tinggi'] & penghasilan['rendah'], prioritas['sangat_tinggi']),
    ctrl.Rule(ipk['sedang'] & penghasilan['rendah'], prioritas['tinggi']),
    ctrl.Rule(ipk['tinggi'] & penghasilan['sedang'], prioritas['tinggi']),
    ctrl.Rule(ipk['rendah'] | penghasilan['tinggi'], prioritas['rendah']),
    ctrl.Rule(ipk['sedang'] & penghasilan['sedang'], prioritas['sedang'])
]

control_sys = ctrl.ControlSystem(rules)
simulation = ctrl.ControlSystemSimulation(control_sys)

x_ipk = np.linspace(0, 4, 30)
y_gaji = np.linspace(0, 10000000, 30) 
X, Y = np.meshgrid(x_ipk, y_gaji)
Z = np.zeros_like(X)

for i in range(30):
    for j in range(30):
        simulation.input['IPK'] = X[i, j]
        simulation.input['Penghasilan_Ortu'] = Y[i, j]
        simulation.input['Prestasi'] = 50 
        simulation.input['Kondisi_Rumah_CNN'] = 50
        simulation.input['Semester'] = 4
        simulation.input['Keaktifan_Organisasi'] = 50
        
        try:
            simulation.compute()
            Z[i, j] = simulation.output['Prioritas_Beasiswa']
        except:
            Z[i, j] = 50 

# Plot 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

ax.set_xlabel('IPK')
ax.set_ylabel('Penghasilan (Rp)')
ax.set_zlabel('Prioritas Beasiswa')
ax.set_title('Surface Plot: Pengaruh IPK dan Penghasilan terhadap Prioritas')
fig.colorbar(surf, shrink=0.5, aspect=5)

save_path_3d = os.path.join(OUTPUT_DIR, "8_surface_plot_3d.png")
plt.savefig(save_path_3d, dpi=300)
plt.close()

print(f"  [OK] Tersimpan: {save_path_3d}")
print("\nSemua visualisasi selesai! Cek folder 'gambar_laporan'.")