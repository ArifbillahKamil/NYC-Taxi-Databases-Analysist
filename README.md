# 🚖 NYC Taxi Fare Prediction with PySpark

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
Proyek ini adalah implementasi *end-to-end* pipeline **Big Data & Machine Learning** untuk memprediksi tarif taksi di New York City.

Menggunakan **Apache Spark (PySpark)**, proyek ini memproses data transaksi taksi, melakukan pembersihan data (*data cleaning*), rekayasa fitur (*feature engineering*), serta melatih dan mengoptimalkan model Machine Learning (Linear Regression & Random Forest) untuk mengestimasi biaya perjalanan.

Proyek ini disusun untuk memenuhi Tugas Akhir mata kuliah **Kecerdasan Artifisial (Big Data & AI)**.

## 📂 Dataset
Dataset yang digunakan berasal dari kompetisi Kaggle: **New York City Taxi Fare Prediction**.
* **Sumber:** Kaggle
* **Fitur Utama:** `pickup_datetime`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `passenger_count`.
* **Target:** `fare_amount` (Tarif dalam USD).

## 🛠️ Tech Stack
* **Core:** Python, PySpark (Spark SQL & MLlib)
* **Data Manipulation:** Pandas, NumPy (untuk simulasi & visualisasi)
* **Visualization:** Matplotlib, Seaborn

## ⚙️ Project Pipeline

### 1. Data Loading & Cleaning
* Memuat data dengan skema eksplisit (`StructType`) untuk efisiensi memori.
* **Filtering Outliers:** Menghapus data anomali seperti:
    * Tarif negatif (`fare_amount < 0`).
    * Koordinat GPS di luar wilayah NYC.
    * Jumlah penumpang yang tidak logis (0 atau >6).

### 2. Feature Engineering
Mengubah data mentah menjadi fitur yang bermakna bagi model:
* **Distance Calculation:** Menghitung jarak tempuh (*Haversine Distance*) berdasarkan koordinat lintang/bujur.
* **Time Extraction:** Mengekstrak informasi waktu (`hour`, `day_of_week`, `month`, `year`) dari timestamp.
* **Coordinate Differences:** Menghitung selisih absolut lintang dan bujur.

### 3. Modeling
Kami membandingkan dua pendekatan algoritma:
1.  **Linear Regression:** Sebagai *baseline* model untuk menangkap hubungan linier (jarak vs tarif).
2.  **Random Forest Regressor:** Sebagai model utama untuk menangkap pola non-linier yang kompleks (misal: jam sibuk, lokasi spesifik).

### 4. Hyperparameter Tuning
Menggunakan `CrossValidator` dan `ParamGridBuilder` untuk mengoptimalkan performa Random Forest:
* **NumTrees:** 20 vs 50
* **MaxDepth:** 5 vs 10

## 📊 Results & Evaluation

Evaluasi dilakukan menggunakan data uji (20% split) dengan metrik **RMSE** (Root Mean Squared Error) dan **R²** (R-Squared).

| Model | RMSE (Lower is Better) | R² (Higher is Better) |
| :--- | :--- | :--- |
| **Linear Regression** | $4.32 | 78.9% |
| **Random Forest (Default)** | $4.16 | 79.4% |
| **Random Forest (Tuned)** | **$4.01** | **80.8%** |

> **Kesimpulan Teknis:** Model Random Forest yang di-tuning memberikan performa terbaik dengan kemampuan menjelaskan 80.8% variasi harga.

## 💡 Key Insights & Analysis

Melalui analisis mendalam terhadap model (`featureImportances` dan `Error Analysis`), kami menemukan:

1.  **Distance is King:** Fitur `distance_km` adalah faktor paling dominan (>60% importance). Ini mengonfirmasi bahwa tarif taksi sangat bergantung pada jarak tempuh (pola linier).
2.  **The "Traffic" Blindspot:** Analisis error menunjukkan bahwa model secara konsisten **memprediksi terlalu rendah (underpredict)** pada jam-jam sibuk (08:00 & 17:00-18:00).
3.  **Recommendation:** Model saat ini "buta" terhadap kemacetan. Untuk menurunkan RMSE di bawah $4.00, diperlukan penambahan data eksternal seperti **data lalu lintas real-time** atau **data cuaca**.

## 📈 Visualizations
Proyek ini menyertakan visualisasi data untuk memvalidasi temuan:
* **Histogram:** Distribusi tarif (bukti kebersihan data).
* **Scatter Plot:** Korelasi Jarak vs Tarif (bukti pola linier).
* **Bar Chart:** Rata-rata Error per Jam (bukti dampak jam sibuk/kemacetan).

## 🚀 How to Run
1.  Pastikan Python dan PySpark terinstal.
2.  Clone repository ini.
3.  Letakkan file `train.csv` di direktori root.
4.  Jalankan notebook `.ipynb` menggunakan Jupyter Notebook, VS Code, atau Google Colab.

```bash
pip install pyspark pandas matplotlib seaborn numpy
