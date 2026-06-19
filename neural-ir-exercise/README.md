# Tugas Lab: Neural Information Retrieval (IR) & Extractive QA

Repository ini berisi materi latihan dan starter code untuk mata kuliah **Advanced Information Retrieval**, yang berfokus pada pengolahan data judgement multi-annotator (FiRA), neural re-ranking menggunakan modern Transformer models (BERT Cross-Encoder), dan Extractive Question Answering (QA).

**Dosen Pengampu:** Zico Pratama Putra

---

## 📂 Struktur Direktori

Berikut adalah struktur folder yang direkomendasikan untuk diserahkan oleh mahasiswa:

```text
neural-ir-exercise/
├── data/
│   ├── fira_raw_judgements.tsv         # Dataset mentah multi-annotator FiRA (diberikan)
│   ├── fira-2022.tuples.tsv            # Dataset pasangan query-passage untuk evaluasi (diberikan)
│   ├── fira-2022.baseline-qrels.tsv    # Baseline qrels berdasarkan majority vote (diberikan)
│   └── fira_aggregated.qrels           # Qrels hasil agregasi kustom mahasiswa (dihasilkan setelah Part 1)
├── src/
│   ├── judgement_aggregation.py        # Starter code untuk Part 1 (diberikan)
│   ├── bert_cross_encoder.py           # Starter code untuk Part 2 (diberikan)
│   └── extractive_qa.py                # Implementasi Part 3 (dibuat oleh mahasiswa)
├── requirements.txt                    # Dependensi library Python (diberikan)
├── Laporan_Kelompok.pdf                # Laporan akhir kelompok (dikumpulkan mahasiswa)
└── README.md                           # Dokumentasi ini
```

> [!IMPORTANT]
> Silakan buat folder `src/` dan pindahkan file python (`judgement_aggregation.py`, `bert_cross_encoder.py`) ke dalam folder tersebut untuk kerapian proyek Anda sesuai rekomendasi di atas.

---

## 🛠️ Persiapan Lingkungan (Setup Environment)

Gunakan Python 3.8+ (disarankan menggunakan virtual environment seperti `venv` atau `conda`).

1. **Instalasi Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Kebutuhan Hardware:**
   * Untuk menjalankan pelatihan model BERT Cross-Encoder di Part 2, sangat disarankan menggunakan **NVIDIA GPU** dengan CUDA. 
   * Jika tidak memiliki GPU lokal, Anda dapat menggunakan layanan berbasis cloud gratis seperti **Google Colab**.

---

## 📝 Deskripsi Tugas Lab

Tugas Lab ini terdiri dari 4 bagian utama:

### Part 1: FiRA Judgement Aggregation (20 Poin)
Mahasiswa diberikan dataset raw judgements [data/fira_raw_judgements.tsv](data/fira_raw_judgements.tsv) yang berisi penilaian mentah multi-annotator per pasangan query-document. Tugas Anda adalah:
1. Menganalisis data mentah tersebut. Setiap baris mencakup `query_id`, `doc_id`, `judgement` (skor relevansi), `confidence` annotator, `annotator_id`, dan `duration_ms` (durasi waktu pengerjaan).
2. Mengimplementasikan strategi agregasi yang lebih cerdas dibandingkan majority voting sederhana pada file `src/judgement_aggregation.py`. Ide yang bisa dicoba:
   * Pembobotan suara (weighted voting) berdasarkan *confidence score* annotator.
   * Eliminasi annotator yang tidak konsisten atau terlalu cepat menilai (*low-confidence / outlier removal*).
   * Menghitung nilai median dengan threshold tertentu saat terjadi perbedaan pendapat yang tinggi (*high std deviation*).
3. Jalankan skrip agregasi untuk menghasilkan file qrels baru format TREC di `data/fira_aggregated.qrels`.

### Part 2: Neural Re-Ranking dengan BERT Cross-Encoder (40 Poin)
Mahasiswa diminta mengimplementasikan dan melatih/mengevaluasi model BERT Cross-Encoder untuk melakukan re-ranking dokumen hasil pencarian tahap pertama (BM25).
1. Gunakan starter code di `src/bert_cross_encoder.py` yang memanfaatkan library Hugging Face Transformers.
2. Lengkapi fungsi pelatihan (`train`) dengan *pairwise loss* atau *classification loss* menggunakan dataset MS MARCO.
3. Evaluasi performa re-ranker Anda pada data FiRA menggunakan file pasangan evaluasi [data/fira-2022.tuples.tsv](data/fira-2022.tuples.tsv).
4. Bandingkan performa re-ranker Anda menggunakan dua qrels berbeda:
   * **Baseline Qrels**: [data/fira-2022.baseline-qrels.tsv](data/fira-2022.baseline-qrels.tsv) (Majority vote)
   * **Aggregated Qrels**: Hasil agregasi kustom Anda dari Part 1 (`data/fira_aggregated.qrels`)
5. Hitung metrik evaluasi: **MRR@10**, **NDCG@10**, dan **Precision@10**.

### Part 3: Extractive Question Answering (QA) (20 Poin)
Gunakan model Question Answering ekstraktif untuk menemukan jawaban spesifik (span) pada paragraf relevan teratas hasil re-ranking.
1. Gunakan model QA pre-trained yang direkomendasikan: `deepset/roberta-base-squad2`.
2. Jalankan inferensi QA pada passage/dokumen peringkat pertama hasil re-ranker BERT untuk setiap query.
3. Bandingkan jawaban yang dihasilkan model dengan *gold answer* (kunci jawaban) yang ada di dataset. Evaluasi overlap jawaban menggunakan metrik **F1-Score** dan **Exact Match (EM)**.

### Part 4: Laporan Kelompok (20 Poin)
Tulis laporan akhir dalam format PDF (maksimal 8–10 halaman) yang memuat:
1. Pendahuluan dan Metodologi Agregasi (Part 1).
2. Desain eksperimen, arsitektur BERT Cross-Encoder, dan kurva pelatihan (Part 2).
3. Hasil perbandingan metrik evaluasi (MRR, NDCG, P@10) antara baseline qrels vs agregasi kustom Anda.
4. Analisis performa Extractive QA beserta contoh kasus keberhasilan dan kegagalan ekstraksi (Part 3).
5. Pembagian tugas masing-masing anggota kelompok secara transparan.

---

## 🚀 Cara Menjalankan Starter Code

### Menjalankan Agregasi Judgement (Part 1)
```bash
python src/judgement_aggregation.py
```
Skrip ini akan memuat file raw judgements, mengagregasinya sesuai strategi Anda, dan menyimpan hasilnya ke format qrels di `data/fira_aggregated.qrels`.

### Menjalankan BERT Cross-Encoder (Part 2)
```bash
python src/bert_cross_encoder.py
```
Skrip ini memuat model pre-trained `cross-encoder/ms-marco-MiniLM-L-6-v2`, melakukan inferensi re-ranking pada contoh query, dan mencetak skor relevansi dokumen.

---
Selamat Mengerjakan! Jika ada kendala teknis atau pertanyaan, silakan diskusikan di forum kelas atau melalui GitHub Issues di repository kelompok masing-masing.
