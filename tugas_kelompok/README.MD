# NYC Rat Sighting — Prediksi Durasi Penanganan Laporan

Proyek machine learning untuk memprediksi durasi penanganan laporan tikus
di New York City berdasarkan data pengaduan warga melalui sistem NYC 311.

## Struktur File

├── code/
│   ├── preprocessing.ipynb            # Pembersihan & transformasi data
│   └── model.ipynb                    # Training, evaluasi & stress test
├── data/
│   ├── Rat_Sightings.csv              # Dataset mentah dari NYC 311
│   ├── Modelling_Ready.csv            # Output preprocessing → input modeling
│   └── dataset_noise_stress_city.csv  # Dataset untuk stress test
├── Laporan/
│   └── Laporan_ML_Kelompok3.pdf       # Laporan akhir proyek
├── Log_Experiment/
│   └── Log_Experiment.docx            # Catatan hasil eksperimen & metrik
├── Log_Prompt/
│   └── Log_Prompt.docx                # Catatan prompt yang digunakan
└── README.md

## Requirements

pip install pandas numpy scikit-learn xgboost

## Alur Pengerjaan

1. Jalankan `code/preprocessing.ipynb` — membersihkan Rat_Sightings.csv
   dan menghasilkan Modelling_Ready.csv
2. Jalankan `code/model.ipynb` — training, evaluasi, dan stress test
   menggunakan Modelling_Ready.csv dan dataset_noise_stress_city.csv

## Model & Hasil

| Model             | R2 (Split) | R2 (KFold)          |
|-------------------|------------|---------------------|
| Linear Regression | 0.1192     | 0.1125 ± 0.0053     |
| ElasticNet        | 0.1149     | 0.1102 ± 0.0039     |
| **XGBoost**       | **0.3458** | **0.3288 ± 0.0163** |

Model terbaik: **XGBoost** — akurasi tertinggi & paling tahan noise (ΔR2 = 0.0157)

## Catatan

Pastikan notebook dijalankan sesuai urutan — preprocessing.ipynb
harus selesai terlebih dahulu sebelum model.ipynb dijalankan.