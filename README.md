# Analisis Sentimen Pemindahan Ibu Kota Negara (IKN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Proyek ini bertujuan untuk menganalisis sentimen publik (positif dan negatif) terhadap pemindahan Ibu Kota Negara (IKN) Indonesia ke Kalimantan Timur. Menggunakan teknik *Machine Learning* dan *Natural Language Processing* (NLP) untuk mengklasifikasikan opini dari media sosial (Twitter/X).

## 📋 Daftar Isi
- [Fitur Utama](#-fitur-utama)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Struktur Project](#-struktur-project)

## 🌟 Fitur Utama

1.  **Data Preprocessing**: Pembersihan teks (menghapus URL, mention, hashtag), *case folding*, dan *stopword removal* khusus Bahasa Indonesia.
2.  **Feature Extraction**: Menggunakan TF-IDF (*Term Frequency-Inverse Document Frequency*) untuk mengubah teks menjadi representasi numerik.
3.  **Handling Imbalanced Data**: Penerapan **SMOTE** (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan jumlah data positif dan negatif.
4.  **Modeling**: Komparasi dua algoritma Machine Learning:
    * Support Vector Machine (SVM)
    * Random Forest Classifier
5.  **Analisis Topik**: Kategorisasi opini berdasarkan kata kunci (Ekonomi, Lingkungan, Infrastruktur, dll).
6.  **Visualisasi**: Grafik distribusi sentimen dan *word importance* (kata kunci yang paling berpengaruh).

## 🛠 Teknologi yang Digunakan

* **Python**: Bahasa pemrograman utama.
* **Pandas & NumPy**: Manipulasi dan analisis data.
* **Scikit-learn**: Pembuatan model ML (SVM, Random Forest) dan evaluasi.
* **Sastrawi**: *Stemming* dan *Stopword removal* khusus Bahasa Indonesia.
* **Imbalanced-learn**: Implementasi SMOTE.
* **Matplotlib & Seaborn**: Visualisasi data.

## 📂 Struktur Project

```text
ikn-sentiment-analysis/
│
├── data/
│   └── ikn.csv               # Dataset mentah (tweet & label sentiment)
│
├── notebooks/
│   └── IKN_Sentiment_Analysis.ipynb  # Main notebook (Preprocessing hingga Evaluasi)
│
├── README.md                 # Dokumentasi proyek
└── requirements.txt          # Daftar library yang dibutuhkan
