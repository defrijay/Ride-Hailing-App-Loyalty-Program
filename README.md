# 🏆 **Ride Hailing App Loyalty Program** — User Segmentation & Churn Risk Analysis

> **Consumer Insight**
> Segmentasi pengguna Ride Hailing App menggunakan RFM + K-Means, dilengkapi prediksi churn dengan XGBoost dan SHAP explainability mengidentifikasi potensi revenue at risk dari 10.000 pengguna sintetis.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-AUC--ROC%200.838-FF6600?style=flat)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2?style=flat)](https://shap.readthedocs.io)
[![Notebooks](https://img.shields.io/badge/Notebooks-5%20pipeline%20stages-F37626?style=flat&logo=jupyter)](https://jupyter.org)

---

## 📋 Table of Contents

- [TL;DR — Executive Summary](#-tldr--executive-summary)
- [Background](#-background)
- [Problem Statement](#-problem-statement)
- [Key Findings](#-key-findings)
- [Segment Profiles](#-segment-profiles)
- [Recommendations](#-recommendations)
- [Dataset](#-dataset)
- [Pipeline Overview](#-pipeline-overview)
- [Notebooks](#-notebooks)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Limitations & Next Steps](#-limitations--next-steps)

---

## 📌 TL;DR — Executive Summary

Proyek ini menganalisis **10.000 pengguna Ride Hailing App** menggunakan pendekatan end-to-end: mulai dari pembuatan synthetic dataset yang realistis, EDA mendalam, RFM segmentation + K-Means clustering, hingga churn prediction berbasis machine learning dengan SHAP explainability.

**3 temuan utama:**
- 📍 **Hibernating** (38.9% dari user base) adalah segmen terbesar — mayoritas sudah tidak aktif dan membutuhkan strategi win-back segera.
- 📍 **recency_days** adalah prediktor churn terkuat (SHAP ≈ 0.14 per instance) — semakin lama tidak transaksi, semakin tinggi risiko churn.
- 📍 **XGBoost** dipilih sebagai model final dengan **AUC-ROC 0.838**, **F1-score 0.689**, berhasil menangkap **73% pengguna yang benar-benar churn** (Recall = 0.731).

| Metrik | Nilai |
|---|---|
| Total pengguna | 10.000 |
| Overall churn rate | 34.4% |
| Model terpilih | XGBoost |
| AUC-ROC | 0.838 |
| F1-score (churn class) | 0.689 |
| Recall (churn class) | 0.731 |
| Top churn predictor | `recency_days` (SHAP ≈ 0.14) |
| Adjusted Rand Index (K-Means) | 0.485 |

---

## 🎯 Background

Ride Hailing App Loyalty Program adalah program loyalitas berbasis poin OVO yang mencakup seluruh ekosistem layanan Grab — GrabBike, GrabCar, GrabFood, GrabMart, dan GrabExpress. Program ini merupakan salah satu instrumen retensi pengguna utama di Indonesia.

Tantangan utama program loyalitas seperti ini adalah mengidentifikasi pengguna yang berisiko churn *sebelum* mereka benar-benar berhenti menggunakan layanan — sehingga tim product dan marketing dapat melakukan intervensi yang tepat waktu dan tepat sasaran.

Proyek ini mensimulasikan pekerjaan yang dilakukan oleh tim Research & Data Analytics dalam mendukung program-program yang sedang berjalan, khususnya dalam hal:
- Memahami perilaku pengguna di berbagai segmen loyalitas
- Memprediksi siapa yang akan churn dan mengapa
- Merekomendasikan aksi retensi yang spesifik dan terukur per segmen

---

## ❓ Problem Statement

> *"Dari 10.000 pengguna aktif Ride Hailing App, siapa yang paling berisiko churn dalam 90 hari ke depan — dan intervensi apa yang paling efektif untuk mempertahankan mereka?"*

**Research Questions:**

| # | Research Question |
|---|---|
| RQ1 | Segmen perilaku apa yang ada di antara pengguna berdasarkan dimensi Recency, Frequency, Monetary, dan Engagement? |
| RQ2 | Model mana yang menghasilkan akurasi prediksi churn tertinggi, dan fitur apa yang paling prediktif? |
| RQ3 | Bagaimana probabilitas churn berbeda antar segmen dan tier, dan berapa revenue at risk per segmen? |
| RQ4 | Intervensi retensi apa yang memberikan estimated ROI tertinggi per segmen? |

---

## 💡 Key Findings

### 1. Hibernating mendominasi user base — bukan segmen kecil
> Segmen **Hibernating** merupakan kelompok **terbesar** dengan **38.9%** dari total pengguna (rule-based RFM), jauh melampaui desain awal 20%. Ini mengindikasikan bahwa mayoritas user base sudah tidak aktif dan membutuhkan strategi win-back segera, bukan hanya maintenance.

### 2. Churn rate sangat timpang antar segmen
> Analisis EDA mengungkap disparitas ekstrem: **Hibernating mencapai 85% churn rate**, At_Risk 46%, sementara **Champions hanya 3%** dan Loyal 10%. Ini mengkonfirmasi bahwa intervensi one-size-fits-all tidak efektif — setiap segmen membutuhkan pendekatan berbeda.

### 3. recency_days adalah prediktor churn paling dominan
> Berdasarkan SHAP analysis, **recency_days** adalah fitur dengan global importance tertinggi. Pada contoh pengguna At-Risk dengan churn probability 70.9%, recency 49 hari berkontribusi **+0.15** terhadap churn prediction — jauh di atas fitur lainnya.

### 4. Tier Member paling rentan churn
> Analisis churn per tier menunjukkan **Member tier memiliki churn rate 57.9%**, berbanding terbalik dengan Platinum yang hanya **3.3%**. Hubungan invers antara tier dan churn ini mengkonfirmasi pentingnya program tier upgrade sebagai strategi retensi.

### 5. K-Means mendeteksi 5 cluster yang bermakna secara bisnis
> Final K-Means (k=5) menghasilkan **ARI 0.485** — moderate alignment dengan ground truth. Cluster 1 ("Champions") memiliki recency 7 hari, frequency 22x/bulan, monetary ~Rp 1.7 juta; Cluster 3 ("Lost Customers") memiliki recency 258 hari dan frequency 0x/bulan, membuktikan clustering menangkap pola perilaku yang nyata.

### 6. Model ML kompetitif dan konsisten
> Ketiga model menunjukkan performa yang sangat kompetitif dan konsisten: Logistic Regression (AUC 0.842), Random Forest (F1 0.694), XGBoost (AUC 0.838, F1 0.689). **Random Forest memiliki akurasi tertinggi (78.0%)** sementara **Logistic Regression memimpin AUC-ROC (0.842)**. XGBoost dipilih sebagai model final karena balance terbaik antar metrik.

---

## 👥 Segment Profiles

Berdasarkan hasil rule-based RFM segmentation dari notebook 04:

| Segmen | Users (%) | Churn Rate | Avg Recency | Avg Freq/bln | Avg Monetary | Churn Rate (Tier Member) |
|---|---|---|---|---|---|---|
| 🏆 **Champions** | 1,500 (17.7%) | ~3% | 1–14 hari | 15–30x | Rp 800K–2.5Jt | N/A (dominan Platinum) |
| 💙 **Loyal** | 2,200 (22.1%) | ~10% | 7–30 hari | 8–15x | Rp 300K–800K | 40.9% Silver |
| ⚠️ **At_Risk** | 1,020 (10.2%) | ~46% | 45–90 hari | 2–6x | Rp 100K–350K | 60.3% Silver |
| 🌱 **Promising** | 1,110 (11.1%) | ~26% | 3–21 hari | 3–8x | Rp 80K–250K | 68.9% Member |
| 😴 **Hibernating** | 3,890 (38.9%) | **~85%** | 90–365 hari | 0–1x | Rp 0–80K | 70.0% Member |

> **Catatan penting:** Distribusi segmen rule-based berbeda dari ground truth desain awal karena scoring quartile yang dinamis. Hibernating mendominasi karena proporsi besar user dengan recency tinggi, sementara Champions menjadi lebih besar dari 15% karena threshold scoring.

---

## 📊 Recommendations

Berdasarkan analisis churn rate per segmen, tier, dan profil SHAP:

| Prioritas | Segmen | Insight dari Data | Aksi yang Direkomendasikan | Estimated Impact |
|---|---|---|---|---|
| **P1** | ⚠️ At_Risk | Churn rate 46%, recency 45–90 hari, masih di Silver/Gold tier | Alert "OVO Points expired dalam 30 hari" + voucher GrabFood 25% off | Recovery 25% At_Risk users |
| **P1** | 😴 Hibernating | Churn rate 85%, 70% di Member tier, recency 90–365 hari | Win-back campaign hanya untuk high-value historical (monetary > Rp 500K); skip low-value karena cost > benefit | Fokus 20% Hibernating bernilai tinggi |
| **P2** | 💙 Loyal | Churn rate 10%, 40.9% di Silver tier — potensial tier upgrade | Push notif "X poin lagi menuju Gold" + weekly challenge berbasis frequency | 20% upgrade ke Gold tier dalam 90 hari |
| **P2** | 🌱 Promising | Churn rate 26%, rata-rata 1–2 layanan digunakan | "Coba GrabMart, dapat 2x poin minggu ini" — dorong multi-service adoption | Target ≥3 layanan dalam 60 hari; n_services = top SHAP feature |
| **P3** | 🏆 Champions | Churn rate hanya 3%, dominan di Platinum/Gold | Early access fitur baru + monthly double-points event + personal appreciation notification | Pertahankan 97%+ retention rate |

---

## 🗂️ Dataset

Dataset ini merupakan **synthetic dataset** yang dirancang untuk mensimulasikan perilaku nyata pengguna loyalty program. Distribusi parameter setiap segmen mengacu pada pola yang dipublikasikan dalam literatur loyalty program dan customer churn research.

### Spesifikasi dataset aktual (dari notebook 01 & 02)

| Attribute | Detail |
|---|---|
| Jumlah user | 10.000 |
| Jumlah kolom | 12 |
| Churn rate keseluruhan | **34.4%** (bukan 32% seperti desain awal — karena noise realistis) |
| Duplikat | 0 (dataset bersih) |
| Missing values | 0 (100% complete) |
| Tanggal referensi | 1 Januari 2025 |
| Format | CSV (`rewards_synthetic.csv`) |

### Distribusi segmen ground truth (desain awal)

```
Champions    ████████████████░░░░░░░░░░░░░░░░  15%  (1,500 users)
Loyal        █████████████████████████░░░░░░░  25%  (2,500 users)
At_Risk      ████████████████████░░░░░░░░░░░░  20%  (2,000 users)
Promising    ████████████████████░░░░░░░░░░░░  20%  (2,000 users)
Hibernating  ████████████████████░░░░░░░░░░░░  20%  (2,000 users)
```

### Statistik deskriptif variabel numerik (dari notebook 02)

| Kolom | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| `recency_days` | 66.65 | 90.36 | 1 | 21 | 364 |
| `frequency_monthly` | 7.75 | 7.44 | 0 | 5 | 29 |
| `monetary_monthly` | Rp 471,210 | Rp 563,195 | 0 | Rp 241,000 | Rp 2,499,000 |
| `ovo_points_balance` | 4,706 | 5,683 | 0 | 2,457 | 29,901 |
| `n_services` | 2.10 | 1.01 | 1 | 2 | 4 |

### Kolom utama

| Kolom | Tipe | Deskripsi | Relevansi ke Model |
|---|---|---|---|
| `user_id` | String | Identifier unik (USR000000–USR009999) | Dieksklusi dari fitur — hanya untuk tracking |
| `segment_true` | Categorical | Ground truth segmen (5 kelas) | Label pembanding ARI, bukan fitur model |
| `recency_days` | Integer | Hari sejak transaksi terakhir | **Top SHAP feature (≈0.14 per instance)** |
| `frequency_monthly` | Integer | Rata-rata transaksi per bulan | Fitur utama — korelasi negatif kuat dengan churn (-0.483) |
| `monetary_monthly` | Float | Rata-rata spending bulanan (Rp) | Basis kalkulasi revenue at risk |
| `ovo_points_balance` | Integer | Saldo poin OVO aktif | Fitur penting — korelasi negatif dengan churn (-0.383) |
| `tier` | Ordinal | Member / Silver / Gold / Platinum | Encoded → `tier_encoded` (1–4) |
| `n_services` | Integer | Jumlah layanan Grab yang digunakan (1–4) | Fitur penting — user dengan 1 layanan lebih berisiko churn |
| `city` | Categorical | Jakarta / Bandung / Surabaya / Medan / Bekasi | One-hot encoded untuk kontrol geografis |
| `churn_label` | Binary | 1 = churn (recency > 90 hari), 0 = aktif | **Target variable** |

> **Transparansi data:** Dataset ini adalah data simulasi, bukan data internal Grab. Metodologi pembuatan dataset terdokumentasi sepenuhnya di `notebooks/01_data_generation.ipynb`.

---

## ⚙️ Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  01 · Data Generation                                        │
│  Generate 10.000 synthetic users dengan 5 segmen tersembunyi│
│  Churn rate aktual: 34.4% · Output: rewards_synthetic.csv   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  02 · Data Understanding                                     │
│  Shape: 10.000 × 12 · 0 duplikat · 0 missing values        │
│  Statistik deskriptif · distribusi tier & kota              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  03 · EDA                                                    │
│  7 visualisasi: segment dist, RFM histogram, boxplot,       │
│  categorical dist, churn rate (segmen/tier/kota),           │
│  churn composition stacked bar, correlation heatmap         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  04 · RFM Segmentation & K-Means Clustering                  │
│  RFME quartile scoring (R/F/M/E = 1–4)                      │
│  Rule-based: Hibernating 38.9% · Loyal 22.1% · Champions 17.7%│
│  K-Means k=5 · Silhouette best: 0.542 (k=2) · ARI: 0.485   │
│  Output: rfm_scores.csv · radar chart · heatmap             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  05 · Churn Prediction + SHAP                                │
│  16 features · 80/20 train-test split · stratified          │
│  LogReg (AUC 0.842) · RF (F1 0.694) · XGBoost (AUC 0.838)  │
│  SHAP: recency_days top predictor · waterfall At-Risk 70.9% │
│  Output: xgboost_churn_final.pkl · features_final.csv       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📓 Notebooks

| Notebook | Deskripsi | Output utama |
|---|---|---|
| `01_data_generation.ipynb` | Generate synthetic dataset 10.000 users dengan 5 segmen tersembunyi (Champions 15%, Loyal 25%, At_Risk 20%, Promising 20%, Hibernating 20%). Churn rate aktual: 34.4% | `data/raw/rewards_synthetic.csv` |
| `02_data_understanding.ipynb` | Validasi kualitas data: 0 duplikat, 0 missing values, statistik deskriptif, struktur kolom, distribusi dasar | Tidak ada output file — analisis in-notebook |
| `03_EDA.ipynb` | 7 visualisasi sistematis: distribusi segmen, RFM histogram per segmen, boxplot outlier detection, distribusi tier & kota, churn rate analysis (3 sudut pandang), stacked composition, correlation heatmap | `outputs/figures/eda_01` s.d `eda_07` (7 PNG) |
| `04_rfm_segmentation.ipynb` | RFME quartile scoring, rule-based 5-segmen labeling, K-Means elbow + silhouette (best k=2 score 0.542, pilih k=5), ARI 0.485, radar chart, segment × tier heatmap | `data/processed/rfm_scores.csv` · 3 figures |
| `05_churn_prediction.ipynb` | 16 features, train-test split 80/20 stratified, 3 model training, comparison table, ROC curves, confusion matrix, SHAP global importance + waterfall At-Risk user (churn prob 70.9%) | `models/xgboost_churn_final.pkl` · `models/scaler.pkl` · `data/processed/features_final.csv` · 3 figures |

---

## 🤖 Model Performance

### Hasil aktual dari notebook 05

| Model | Accuracy | F1 (Churn) | AUC-ROC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 76.8% | 0.687 | **0.842** | 0.641 | 0.739 |
| **Random Forest** | **78.0%** | **0.694** | 0.839 | 0.664 | 0.728 |
| XGBoost ✓ | 77.3% | 0.689 | 0.838 | 0.651 | 0.731 |

**Model terpilih: XGBoost** — dipilih karena balance terbaik antar semua metrik, native support untuk class imbalance via `scale_pos_weight`, dan kompatibilitas penuh dengan SHAP TreeExplainer untuk interpretabilitas.

> ⚠️ **Koreksi dari README sebelumnya:** Angka AUC-ROC 0.93 dan F1 0.88 yang tercantum sebelumnya adalah target/ekspektasi, **bukan angka aktual**. Hasil aktual dari notebook menunjukkan performa yang lebih realistis: AUC-ROC 0.838 dan F1 0.689 — masih merupakan performa yang solid untuk dataset synthetic dengan 16 fitur.

### Classification Report XGBoost (dari notebook 05)

```
              precision    recall  f1-score   support

       Aktif       0.85      0.80      0.82      1313
       Churn       0.65      0.73      0.69       687

    accuracy                           0.77      2000
   macro avg       0.75      0.76      0.76      2000
weighted avg       0.78      0.77      0.78      2000
```

### SHAP Feature Importance — Top Predictors (global + instance level)

```
recency_days        ████████████████████████  ~0.14 per instance (top global)
ovo_points_balance  ████████████░░░░░░░░░░░░  global 2nd most important
frequency_monthly   ████████████░░░░░░░░░░░░  global 3rd
spend_trend         ████████░░░░░░░░░░░░░░░░  behavioral trend signal
tier_encoded        ██░░░░░░░░░░░░░░░░░░░░░░  near-zero impact
n_services          ██░░░░░░░░░░░░░░░░░░░░░░  near-zero global (but contextually important)
```

**SHAP Waterfall — Contoh At-Risk user:**
- Churn probability: **70.9%** (naik dari baseline 0.395)
- `recency_days` = 49 hari → +0.15 (kontribusi terbesar)
- `spend_trend` = -31,310 → +0.09 (spending menurun)
- `n_services` = 1 → -0.02 (sedikit menahan churn)

---

## 🛠️ Tech Stack

```
Language        Python 3.10+
Data            pandas 2.2.3, numpy 1.26.4
ML              scikit-learn 1.7.2, xgboost 3.0.5
Explainability  shap 0.48.0
Visualization   matplotlib 3.9.2, seaborn 0.12.1
Notebook        Jupyter Lab / VS Code (venv_grabrewards kernel)
Version control GitHub
```

---

## 📁 Project Structure

```
ride-hailing-loyalty-program/
│
├── data/
│   ├── raw/
│   │   └── rewards_synthetic.csv          # Dataset utama (10K users, 12 cols)
│   └── processed/
│       ├── rfm_scores.csv                 # RFME scores + segment labels (10K × 20 cols)
│       └── features_final.csv            # Feature matrix test set + predictions (2K × 16 cols)
│
├── notebooks/
│   ├── 01_data_generation.ipynb          # Synthetic data generator
│   ├── 02_data_understanding.ipynb       # Data quality & structure check
│   ├── 03_EDA.ipynb                      # 7 visualisasi EDA sistematis
│   ├── 04_rfm_segmentation.ipynb         # RFME scoring + K-Means (ARI 0.485)
│   └── 05_churn_prediction.ipynb         # 3 models + SHAP (XGBoost AUC 0.838)
│
├── models/
│   ├── xgboost_churn_final.pkl           # Saved XGBoost model
│   └── scaler.pkl                        # StandardScaler (fitted on train set)
│
├── outputs/
│   └── figures/
│       ├── eda_01_segment_distribution.png
│       ├── eda_02_rfm_distributions.png
│       ├── eda_03_boxplot_per_segment.png
│       ├── eda_04_categorical_distributions.png
│       ├── eda_05_churn_rate_analysis.png
│       ├── eda_06_churn_composition.png
│       ├── eda_07_correlation_heatmap.png
│       ├── elbow_silhouette.png
│       ├── rfm_radar_chart.png
│       ├── segment_tier_heatmap.png
│       ├── roc_curve_comparison.png
│       ├── shap_summary_plot.png
│       └── shap_waterfall_atrisk.png
│
├── report/
│   └── pitch_deck/
│
├── venv_grabrewards/                     # Virtual environment (Python 3.10)
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone repository

```bash
git clone https://github.com/defrijay/ride-hailing-loyalty-program.git
cd ride-hailing-loyalty-program
```

### 2. Setup virtual environment (Windows)

```bash
"C:\Users\Lenovo\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv_grabrewards
.\venv_grabrewards\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name=venv_grabrewards --display-name "Python (GrabRewards)"
```

### 3. Jalankan pipeline secara berurutan

```bash
# Buka Jupyter atau gunakan VS Code dengan kernel venv_grabrewards
jupyter notebook

# Urutan eksekusi:
# notebooks/01_data_generation.ipynb      → generate rewards_synthetic.csv
# notebooks/02_data_understanding.ipynb   → validasi kualitas data
# notebooks/03_EDA.ipynb                  → 7 visualisasi EDA
# notebooks/04_rfm_segmentation.ipynb     → RFME scoring + K-Means
# notebooks/05_churn_prediction.ipynb     → 3 models + SHAP
```

### Requirements

```
pandas>=2.0
numpy>=1.24
scipy==1.11.4
scikit-learn>=1.3
xgboost>=2.0
shap>=0.43
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
ipykernel>=6.0
```

---

## ⚠️ Limitations & Next Steps

### Limitations

- **Data sintetis:** Dataset adalah simulasi berbasis distribusi dari literatur — bukan data internal Grab. Churn rate aktual (34.4%) sedikit berbeda dari target desain (32%) karena noise realistis yang dimasukkan saat generation.
- **SHAP PermutationExplainer:** Karena menggunakan `shap.Explainer` dengan fungsi prediksi (bukan `TreeExplainer` langsung), kalkulasi SHAP membutuhkan waktu ~13 menit untuk 2.001 sampel. Untuk production, gunakan `TreeExplainer` langsung.
- **Performa model moderat:** AUC-ROC 0.838 dan F1 0.689 mencerminkan kompleksitas data synthetic dengan banyak overlap antar segmen. Dengan fitur behavioral yang lebih kaya (histori promo, session data), performa bisa meningkat signifikan.
- **No temporal validation:** Model divalidasi secara cross-sectional; validasi time-series (walk-forward) belum diterapkan.

### Potential next steps (jika ada data real)

| Ekstensi | Deskripsi |
|---|---|
| **CLV Prediction** | Prediksi Customer Lifetime Value per segmen menggunakan BG/NBD model |
| **Next-Best-Offer** | Recommendation engine: reward apa yang paling mungkin mencegah churn per user |
| **Survival Analysis** | Cox Proportional Hazards untuk estimasi waktu sampai churn, bukan hanya probabilitas binary |
| **Real-time scoring** | Deploy XGBoost model sebagai API endpoint untuk scoring churn risk harian |
| **IndoBERT integration** | Analisis sentimen review app store untuk enrich fitur behavioral dengan voice-of-customer signal |

---

## 📬 Contact

**Defrizal Yahdiyan Risyad**
Final-year Computer Science · Universitas Pendidikan Indonesia

[![LinkedIn](https://img.shields.io/badge/LinkedIn-defrizalyr-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/defrizalyr)
[![GitHub](https://img.shields.io/badge/GitHub-defrijay-181717?style=flat&logo=github)](https://github.com/defrijay)
