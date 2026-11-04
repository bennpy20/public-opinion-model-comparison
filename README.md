# Analisis Perbandingan Model Deep Learning untuk Klasifikasi Emosi pada Teks Opini Publik Berbahasa Indonesia

Repository ini berisi prototype tugas akhir berjudul **“Analisis Perbandingan Model Deep Learning untuk Klasifikasi Emosi pada Teks Opini Publik Berbahasa Indonesia”**.

Prototype ini membandingkan empat model, yaitu TextCNN, BiLSTM, IndoBERT, dan RoBERTa, dengan Logistic Regression sebagai baseline. Setiap model diuji menggunakan dua skema:

- **Embeddings**: menggunakan tokenisasi kata dan embedding yang dilatih dari data.
- **Tokenizer**: menggunakan tokenizer subword dan embedding pre-trained.

## Dataset

Dataset yang digunakan adalah *Emotion Dataset from Indonesian Public Opinion* (Riccosan dkk.) dengan 7.080 teks dan 6 kelas emosi: `anger`, `fear`, `joy`, `love`, `neutral`, dan `sad`.

Preprocessing meliputi case folding, penghapusan URL, mention, karakter non-print, data duplikat, dan baris kosong. Data kemudian dibagi menjadi train, validation, dan test dengan rasio 70:15:15.

## Struktur Repository

```text
.
├── 1_LoadData_EDA.ipynb
├── 2_Preprocessing.ipynb
├── 3_SplitData.ipynb
├── 04b_baseline_embeddings.ipynb
├── 04c_baseline_tokenizer.ipynb
├── 05a_textcnn_embeddings.ipynb
├── 05b_textcnn_tokenizer.ipynb
├── 06a_bilstm_embeddings.ipynb
├── 06b_bilstm_tokenizer.ipynb
├── 07a_indobert_embeddings.ipynb
├── 07b_indobert_tokenizer.ipynb
├── 08a_roberta_embeddings.ipynb
├── 08b_roberta_tokenizer.ipynb
├── utils_embedding.py
├── data/
└── outputs/
```

Folder `data/` berisi dataset yang sudah diproses, sedangkan `outputs/` menyimpan hasil evaluasi, grafik, dan model yang telah dilatih.

## Cara Menjalankan

Project dijalankan dengan menggunakan GPU T4 Google Colab (disarankan)

Pastikan seluruh library yang dibutuhkan telah tersedia. Jalankan perintah berikut untuk menginstal atau memperbarui library yang diperlukan:

```bash
pip install torch transformers gensim scikit-learn pandas matplotlib
```

Jalankan notebook dengan urutan berikut:

1. `1_LoadData_EDA.ipynb`
2. `2_Preprocessing.ipynb`
3. `3_SplitData.ipynb`
4. Notebook model sesuai skema yang ingin diuji.

Sebelum menjalankan notebook `04c`, `05b`, atau `06b`, pastikan file `utils_embedding.py` tersedia di folder yang sama.

## Evaluasi

Model dievaluasi menggunakan validation set dengan metrik:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix

Test set belum digunakan pada tahap prototype (progress 50%) ini.

## Hasil

Hasil evaluasi setiap model disimpan dalam folder `outputs/` dengan format JSON. Tabel perbandingan akan dilengkapi setelah seluruh eksperimen selesai dijalankan.

## Model

File model hasil training disimpan di Hugging Face karena ukurannya cukup besar untuk diunggah ke GitHub.  
Silakan unduh model melalui tautan berikut:  

[Hugging Face – Link model](https://huggingface.co/ebenpy/public-opinion-model-comparison)

Setelah proses pengunduhan selesai, buat folder models di dalam direktori `outputs/`, sehingga struktur direktori menjadi `outputs/models/`. Selanjutnya, letakkan seluruh file model yang telah diunduh ke dalam folder tersebut agar dapat digunakan oleh program.

## Catatan Progress STA

Repository ini masih berupa prototype (progress 50%). Setiap kombinasi model dijalankan satu kali dengan seed 42 dan belum melalui hyperparameter tuning maupun pengujian signifikansi statistik.
