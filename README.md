# Movie Recommendation System

He thong goi y phim da phuong tien ket hop phan tich hinh anh ap phich (poster) va noi dung van ban (mo ta phim).

## Cac Mo Hinh Su Dung

| Mo hinh | Muc dich |
|---------|----------|
| **ResNet50** (CNN) | Trich xuat dac trung anh poster (2048 chieu) |
| **TF-IDF** | Trich xuat dac trung van ban mo ta phim (500 chieu) |
| **K-Means** (K=20) | Phan cum phim tuong dong |
| **Naive Bayes** (multi-label) | Du doan the loai phim tu anh poster |
| **Apriori** | Kham pha luat ket hop the loai phim |
| **Cosine Similarity** | Tinh do tuong dong de goi y |

## Tech Stack

**Backend:** Python · FastAPI · SQLAlchemy · SQLite · TensorFlow · scikit-learn · mlxtend

**Frontend:** React 18 · Vite · Tailwind CSS · Recharts · Axios

---

## Yeu Cau He Thong

- Python >= 3.9
- Node.js >= 18
- GPU (khuyen nghi cho Module 2, co the dung CPU nhung cham hon)
- RAM >= 8GB

---

## Huong Dan Cai Dat Va Khoi Chay

Có **2 cách** tùy vào bạn là thành viên mới hay muốn tự train lại từ đầu.

---

### Cach A: Nhanh (danh cho teammate — khong can chay notebook)

> Cach nay tai model da train san tu Google Drive, bo qua hoan toan buoc thu thap du lieu va train.

**Buoc 1: Clone va cai thu vien**

```bash
git clone <repository-url>
cd KhaiPha
pip install -r requirements.txt
pip install gdown
```

**Buoc 2: Tai models tu Google Drive (tu dong)**

```bash
python setup_assets.py
```

Script se tai ~135MB gom: `combined_features.npy`, `cnn_features.npy`, `tfidf_matrix.npy`, cac file `.pkl` va `movies_valid.csv`.

**Buoc 3: Chay Backend**

```bash
cd backend
uvicorn app:app --reload --port 8000
```

**Buoc 4: Chay Frontend**

Mo terminal moi:

```bash
cd frontend
npm install
npm run dev
```

Mo trinh duyet: **http://localhost:5173**

---

### Cach B: Tu train lai tu dau (day du)

> Danh cho nguoi muon chay lai toan bo pipeline tu raw data.

**Buoc 1: Clone va cai thu vien**

```bash
git clone <repository-url>
cd KhaiPha
pip install -r requirements.txt
```

**Buoc 2: Lay API Keys**

**Kaggle API** (de tai dataset):
1. Vao https://www.kaggle.com → Account → API → Create New Token
2. Dat file `kaggle.json` vao `C:\Users\<ten>\.kaggle\kaggle.json`

**TMDB API** (de fetch poster):
1. Vao https://www.themoviedb.org → Settings → API → Create
2. Copy **API Key (v3 auth)**

**Buoc 3: Chay cac Jupyter Notebook theo thu tu**

```bash
jupyter notebook
```

| Notebook | Thoi gian | Output |
|----------|-----------|--------|
| `01_data_collection.ipynb` | ~15 phut | `data/processed/movies.csv` |
| `02_feature_extraction.ipynb` | ~10 phut (GPU) / ~2 gio (CPU) | `models/*.npy`, `models/scalers.pkl` |
| `03_ml_models.ipynb` | ~5 phut | `models/kmeans.pkl`, `nb_model.pkl`, `rules.csv` |

> **Luu y notebook 01:** Dien TMDB API key vao cell 1.4:
> ```python
> TMDB_API_KEY = "your_api_key_here"
> ```

**Buoc 4: Chay Backend**

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Kiem tra: Swagger UI tai http://127.0.0.1:8000/docs

**Buoc 5: Chay Frontend**

```bash
cd frontend
npm install
npm run dev
```

Mo trinh duyet: **http://localhost:5173**

---

## Cau Truc Thu Muc

```
KhaiPha/
├── data/
│   ├── raw/                    # Dataset Kaggle (gitignore)
│   └── processed/              # movies.csv sau xu ly (gitignore)
├── models/                     # File .npy, .pkl sau train (gitignore)
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_ml_models.ipynb
├── backend/
│   ├── app.py                  # FastAPI entry point
│   ├── database.py             # SQLAlchemy + init DB
│   ├── recommender.py          # Cosine similarity logic
│   ├── schemas.py              # Pydantic models
│   ├── routers/
│   │   ├── movies.py
│   │   ├── recommend.py
│   │   ├── genres.py
│   │   └── clusters.py
│   └── test_api.py
├── frontend/
│   └── src/
│       ├── api/movieApi.js
│       ├── components/         # 7 React components
│       └── pages/              # 3 trang chinh
├── implement/                  # Tai lieu tong ket tung module
├── requirements.txt
├── plan.md
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Mo ta |
|--------|----------|-------|
| GET | `/api/movies` | Danh sach phim phan trang |
| GET | `/api/movies/search?q=` | Tim kiem phim theo ten |
| GET | `/api/movies/{id}` | Chi tiet phim + cluster + NB prediction |
| GET | `/api/movies/{id}/recommend?alpha=0.5` | Top-10 phim goi y |
| GET | `/api/genres/rules` | Danh sach association rules |
| GET | `/api/clusters` | Thong ke cac cum phim |
| GET | `/api/clusters/{id}/movies` | Phim trong 1 cum |

**Tham so `alpha`** (0.0 → 1.0):
- `alpha=0.0` → Chi dung dac trung van ban (TF-IDF)
- `alpha=0.5` → Can bang poster va van ban (mac dinh)
- `alpha=1.0` → Chi dung dac trung anh poster (CNN)

---

## Ket Qua Mo Hinh

| Mo hinh | Metric | Gia tri |
|---------|--------|---------|
| K-Means | So cum | 20 |
| K-Means | Silhouette Score | -0.003 |
| Naive Bayes | F1-score (micro) | 0.461 |
| Naive Bayes | Recall (micro) | 0.691 |
| Naive Bayes | Hamming Loss | 0.246 |
| Apriori | So luat (lift > 1) | 12 |
| Apriori | Luat manh nhat | Mystery → Thriller (conf=69.5%, lift=2.59) |

---

## Giao Dien

### Trang Chu
- Tim kiem phim realtime voi dropdown goi y
- Loc theo 14 the loai
- Grid phim responsive (2-5 cot)
- Phan trang

### Trang Chi Tiet Phim
- Poster + day du metadata
- Badge **"The loai du doan boi AI"** (Naive Bayes)
- Thanh dieu chinh **alpha** poster/text → cap nhat goi y ngay
- Grid top-10 phim tuong tu voi % similarity

### Trang Kham Pha
- **PCA 2D scatter plot** — 4752 phim mau theo cum, click de xem chi tiet
- **Bar chart** — so luong phim moi cum, click de xem danh sach
- **Bang Association Rules** — mau theo confidence va lift
