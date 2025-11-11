import pandas as pd
import numpy as np


try:
    url = "https://cdn.kelasai.id/dataset_kelayakan_beasiswa.csv"
    df = pd.read_csv(url)
    print("Dataset berhasil dimuat dari URL Kelas AI.")
except Exception as e:
    print(f"Gagal memuat dari URL: {e}")
    print("Mencoba memuat dari file lokal 'dataset_kelayakan_beasiswa.csv'...")
    try:
        df = pd.read_csv('dataset_kelayakan_beasiswa.csv')
        print("Dataset berhasil dimuat dari file lokal.")
    except FileNotFoundError:
        print("ERROR: File 'dataset_kelayakan_beasiswa.csv' tidak ditemukan.")
        print("Silakan unduh dataset dari Kelas AI dan letakkan di folder proyekmu.")
        exit() 

print("\n--- Analisis Statistik Dataset ---")

print("\n=== Analisis IPK ===")
ipk_stats = df['IPK'].describe()
print(ipk_stats)

print("\n=== Analisis Pendapatan Orang Tua ===")
pd.options.display.float_format = '{:,.0f}'.format
pendapatan_stats = df['Pendapatan_Orang_Tua'].describe()
print(pendapatan_stats)

print("\n--- Rekomendasi Batas Fuzzy Berbasis Data ---")

ipk_min = ipk_stats['min']
ipk_q1 = ipk_stats['25%'] 
ipk_median = ipk_stats['50%'] 
ipk_q3 = ipk_stats['75%'] 
ipk_max = ipk_stats['max']

print("\nRekomendasi Fungsi Keanggotaan IPK:")
print(f"* ipk['rendah']  = fuzz.trimf(ipk_ud, [{ipk_min:.2f}, {ipk_min:.2f}, {ipk_q1:.2f}])")
print(f"* ipk['sedang']  = fuzz.trimf(ipk_ud, [{ipk_q1:.2f}, {ipk_median:.2f}, {ipk_q3:.2f}])")
print(f"* ipk['tinggi']  = fuzz.trimf(ipk_ud, [{ipk_q3:.2f}, {ipk_max:.2f}, {ipk_max:.2f}])")

p_min = pendapatan_stats['min']
p_q1 = pendapatan_stats['25%']
p_median = pendapatan_stats['50%']
p_q3 = pendapatan_stats['75%']
p_max = pendapatan_stats['max']

print("\nRekomendasi Fungsi Keanggotaan Penghasilan (Ingat: Rendah = Butuh):")
print(f"* penghasilan['rendah'] = fuzz.trimf(penghasilan_ud, [{p_min:,.0f}, {p_min:,.0f}, {p_q1:,.0f}])")
print(f"* penghasilan['sedang'] = fuzz.trimf(penghasilan_ud, [{p_q1:,.0f}, {p_median:,.0f}, {p_q3:,.0f}])")
print(f"* penghasilan['tinggi'] = fuzz.trimf(penghasilan_ud, [{p_q3:,.0f}, {p_max:,.0f}, {p_max:,.0f}])")