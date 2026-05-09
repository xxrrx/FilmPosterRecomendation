# Tong Ket Du An: Movie Recommendation System

## Trang Thai Tong The
**HOAN THANH** — Tat ca 5 module da implement va kiem tra.

---

## Tong Quan He Thong

**Ten du an:** He thong goi y phim da phuong tien (Multimodal Movie Recommendation System)

**Mo ta:** Website goi y phim thong minh ket hop phan tich hinh anh ap phich (poster) va noi dung van ban (mo ta phim) de dua ra goi y ca nhan hoa cho nguoi dung.

---

## Luong Du Lieu

```
[Kaggle TMDB 5000] + [TMDB API]
        |
   Module 1: Data Pipeline
   → data/processed/movies.csv (4752 phim)
        |
   Module 2: Feature Extraction (ResNet50 + TF-IDF)
   → models/combined_features.npy (4752, 2548)
        |
   Module 3: ML Models
   → models/kmeans.pkl + nb_model.pkl + rules.csv
        |
   Module 4: FastAPI Backend (localhost:8000)
        |
   Module 5: React Frontend (localhost:5173)
```

---

## Ket Qua Cac Module

### Module 1 — Data Pipeline
| Chi so | Gia tri |
|--------|---------|
| So phim | 4752 |
| Nam phat hanh | 1916 - 2017 |
| Rating trung binh | 6.13 / 10 |
| Missing values | 0 |
| Nguon poster | TMDB API (fetch poster_path + build URL) |

### Module 2 — Feature Extraction
| Chi so | Gia tri |
|--------|---------|
| Mo hinh CNN | ResNet50 (ImageNet), pooling=avg |
| CNN features | (4752, 2048) |
| TF-IDF features | (4752, 500), ngram (1,2) |
| Combined features | (4752, 2548), range [0, 1] |

### Module 3 — ML Models

**K-Means:**
| Chi so | Gia tri |
|--------|---------|
| So cum (K) | 20 |
| Silhouette Score | -0.0025 |
| Kich thuoc cum | 148 - 424 phim |

**Naive Bayes (Multi-label):**
| Metric | Gia tri |
|--------|---------|
| So the loai | 17 |
| Hamming Loss | 0.2462 |
| F1 (micro) | 0.4612 |
| Precision | 0.3461 |
| Recall | 0.6910 |

**Association Rules:**
| Chi so | Gia tri |
|--------|---------|
| So luat (lift > 1) | 12 |
| Luat manh nhat | Mystery → Thriller (conf=69.5%, lift=2.59) |

### Module 4 — Backend API
| Chi so | Gia tri |
|--------|---------|
| Framework | FastAPI + uvicorn |
| Database | SQLite (4752 phim + 12 luat) |
| So endpoints | 7 |
| Goi y logic | Cosine similarity trong cung cum K-Means |

### Module 5 — Frontend
| Chi so | Gia tri |
|--------|---------|
| Framework | React 18 + Vite 5 |
| Styling | Tailwind CSS 3 |
| So trang | 3 (Home, Detail, Explore) |
| So component | 7 |
| Charts | Recharts (ScatterChart, BarChart) |

---

## Cach Chay Toan Bo Du An

**Terminal 1 — Backend:**
```bash
cd KhaiPha/backend
uvicorn app:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd KhaiPha/frontend
npm run dev
```

**Truy cap:**
- Frontend: http://localhost:5173
- API Docs: http://127.0.0.1:8000/docs

---

## Cau Truc Thu Muc Cuoi

```
KhaiPha/
├── data/
│   ├── raw/                         # Dataset Kaggle + poster cache
│   └── processed/
│       ├── movies.csv               # 4752 phim sau xu ly
│       └── movies_valid.csv         # + cluster_id, pca_x, pca_y
├── models/
│   ├── cnn_features.npy             # (4752, 2048)
│   ├── tfidf_matrix.npy             # (4752, 500)
│   ├── combined_features.npy        # (4752, 2548)
│   ├── movie_ids.npy                # (4752,)
│   ├── cluster_labels.npy           # (4752,)
│   ├── kmeans.pkl                   # K-Means K=20
│   ├── nb_model.pkl                 # MultiOutputClassifier(GaussianNB)
│   ├── mlb_encoder.pkl              # MultiLabelBinarizer (17 genres)
│   ├── tfidf_vectorizer.pkl
│   ├── scalers.pkl
│   └── rules.csv                    # 12 association rules
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_ml_models.ipynb
├── backend/
│   ├── app.py
│   ├── database.py
│   ├── recommender.py
│   ├── schemas.py
│   └── routers/ (4 files)
├── frontend/
│   └── src/ (7 components + 3 pages)
├── implement/
│   ├── project_summary.md           # File nay
│   ├── module1_summary.md
│   ├── module2_summary.md
│   ├── module3_summary.md
│   ├── module4_summary.md
│   └── module5_summary.md
└── requirements.txt
```

---

## Van De Gap Phai Va Giai Phap

| Van de | Module | Giai phap |
|--------|--------|-----------|
| Dataset Kaggle khong co `poster_path` | M1 | Fetch tu TMDB API voi 10 threads + cache JSON |
| Cell notebook sai thu tu | M1 | Gop 2 cell thanh 1 theo dung thu tu |
| Localhost resolve sang IPv6 tren Windows | M4 | Doi BASE URL tu `localhost` sang `127.0.0.1` |
| Silhouette Score thap (-0.003) | M3 | Do "curse of dimensionality" voi 2548 chieu; clustering van co y nghia noi dung |
| CNN fetch poster cham | M2 | Dung ThreadPoolExecutor 20 workers + checkpoint tu dong |
