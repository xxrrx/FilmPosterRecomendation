# Kế Hoạch Thực Hiện Chi Tiết: Movie Recommendation System

## Nguyên Tắc Thiết Kế

- Mỗi module hoàn thành là **test được ngay**, độc lập với các module sau
- Dữ liệu truyền giữa các module qua **file trung gian** (`.csv`, `.npy`, `.pkl`) — không phụ thuộc runtime
- Backend API có thể test qua **Swagger UI** mà không cần frontend
- Frontend dùng **mock data** để phát triển song song với backend

---

## Tổng Quan Luồng Dữ Liệu

```
[TMDB API + Kaggle]
        ↓
[Module 1] → movies.csv + posters/
        ↓
[Module 2] → cnn_features.npy + tfidf_matrix.npy + combined_features.npy
        ↓
[Module 3] → kmeans.pkl + cluster_labels.npy + nb_model.pkl + rules.csv
        ↓
[Module 4] → FastAPI server (localhost:8000)
        ↓
[Module 5] → React App (localhost:5173)
```

---

## Module 1: Data Pipeline

### Mục tiêu
Thu thập và chuẩn hóa dữ liệu phim, tải ảnh poster về local.

### Công việc

| Bước | Nội dung | Tool |
|------|----------|------|
| 1.1 | Tải TMDB 5000 Movie Dataset từ Kaggle | Kaggle CLI / tay |
| 1.2 | Parse JSON columns (genres, cast, keywords) | pandas |
| 1.3 | Lọc phim có đủ: poster_path, overview, genres | pandas |
| 1.4 | Lấy poster_path từ TMDB API, build URL: `https://image.tmdb.org/t/p/w500/{poster_path}` | requests |
| 1.5 | Chuẩn hóa text: lowercase, remove stopwords, lemmatize overview | nltk |
| 1.6 | Xuất file `data/processed/movies.csv` | pandas |

### Schema `movies.csv`
```
movie_id, title, year, genres, overview_clean, poster_url, rating, vote_count
```
> `poster_url` = `https://image.tmdb.org/t/p/w500/{poster_path}` — không tải về local

### File output
```
data/
├── raw/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
└── processed/
    └── movies.csv              # poster_url đã được build sẵn
```

### Cách test Module 1
```python
# Cuối notebook 01_data_collection.ipynb
import pandas as pd, requests

df = pd.read_csv("data/processed/movies.csv")
print(f"So phim: {len(df)}")                        # >= 4500
print(f"Missing values:\n{df.isnull().sum()}")      # 0 missing
assert df['poster_url'].str.startswith("https://").all()

# Kiem tra poster URL hop le (test 3 phim dau)
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, row in df.head(3).iterrows():
    resp = requests.get(row['poster_url'])
    img = Image.open(BytesIO(resp.content))
    axes[i].imshow(img)
    axes[i].set_title(row['title'])
plt.show()
```

---

## Module 2: Feature Extraction

### Mục tiêu
Trích xuất đặc trưng từ ảnh poster (CNN) và văn bản mô tả (TF-IDF), kết hợp thành vector tổng hợp.

### Công việc

| Bước | Nội dung | Tool |
|------|----------|------|
| 2.1 | Load ResNet50 pretrained (ImageNet), bỏ lớp FC cuối | keras |
| 2.2 | Fetch poster từ URL TMDB, batch process → feature matrix `(N, 2048)` — dùng GPU | tensorflow |
| 2.3 | TF-IDF trên `overview_clean + genres` → matrix `(N, 500)` | scikit-learn |
| 2.4 | Normalize từng loại về [0,1] với MinMaxScaler | scikit-learn |
| 2.5 | Concatenate → `combined_features (N, 2548)` | numpy |
| 2.6 | Lưu tất cả ra file `.npy` và scaler ra `.pkl` | numpy, joblib |

### File output
```
models/
├── cnn_features.npy        # shape (N, 2048)
├── tfidf_matrix.npy        # shape (N, 500)
├── combined_features.npy   # shape (N, 2548)
├── tfidf_vectorizer.pkl    # để transform phim mới
└── scalers.pkl             # MinMaxScaler cho cả 2 loại
```

### Cách test Module 2
```python
# Cuối notebook 02_feature_extraction.ipynb
import numpy as np

cnn = np.load("models/cnn_features.npy")
tfidf = np.load("models/tfidf_matrix.npy")
combined = np.load("models/combined_features.npy")

print(f"CNN features shape: {cnn.shape}")           # (N, 2048)
print(f"TF-IDF shape: {tfidf.shape}")               # (N, 500)
print(f"Combined shape: {combined.shape}")           # (N, 2548)
print(f"Combined range: [{combined.min():.2f}, {combined.max():.2f}]")  # [0, 1]

# Kiem tra similarity: 2 phim cung the loai phai co score cao hon phim khac the loai
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(combined[[0]], combined)[0]
top5_idx = sim.argsort()[-6:-1][::-1]
print("Top 5 phim tuong tu voi phim 0:")
print(df.iloc[top5_idx][['title', 'genres']])
```

---

## Module 3: ML Models

### Mục tiêu
Train 3 mô hình: K-Means phân cụm, Naive Bayes phân loại thể loại, Apriori tìm luật kết hợp.

### 3.1 K-Means Clustering

| Bước | Nội dung |
|------|----------|
| 3.1.1 | Vẽ Elbow Curve với K từ 5 đến 40 để chọn K tối ưu |
| 3.1.2 | Train KMeans(n_clusters=K_optimal) trên combined_features |
| 3.1.3 | Tính Silhouette Score + Davies-Bouldin Index |
| 3.1.4 | **Precompute PCA 2D:** `pca_coords = PCA(n_components=2).fit_transform(combined_features)` |
| 3.1.5 | Lưu `pca_x`, `pca_y`, `cluster_id` vào `movies.csv` (dùng để import vào SQLite) |
| 3.1.6 | Lưu kmeans.pkl + cluster_labels.npy |

**File output:** `models/kmeans.pkl`, `models/cluster_labels.npy`, cột `pca_x/pca_y/cluster_id` trong `movies.csv`

### 3.2 Naive Bayes Classification

| Bước | Nội dung |
|------|----------|
| 3.2.1 | Encode genres thành multi-label binary matrix `(N, n_genres)` — mỗi cột là 1 thể loại (0/1) |
| 3.2.2 | Lọc giữ lại các thể loại có >= 100 phim |
| 3.2.3 | Train/test split 80/20 |
| 3.2.4 | Train `MultiOutputClassifier(GaussianNB())` trên cnn_features |
| 3.2.5 | Đánh giá: Hamming Loss, F1-score (micro + macro), Precision, Recall |
| 3.2.6 | Lưu nb_model.pkl + mlb_encoder.pkl (MultiLabelBinarizer) |

**File output:** `models/nb_model.pkl`, `models/mlb_encoder.pkl`

### 3.3 Association Rules

| Bước | Nội dung |
|------|----------|
| 3.3.1 | Tạo transaction matrix: mỗi phim = 1 row, genres = columns (0/1) |
| 3.3.2 | Apriori(min_support=0.05) |
| 3.3.3 | association_rules(metric="confidence", min_threshold=0.4) |
| 3.3.4 | Lọc lift > 1.0 (chỉ lấy luật có ý nghĩa) |
| 3.3.5 | Lưu rules.csv |

**File output:** `models/rules.csv`

### Cách test Module 3
```python
# test_models.py — chạy độc lập sau khi train xong
import joblib, numpy as np, pandas as pd
from sklearn.metrics import silhouette_score

combined = np.load("models/combined_features.npy")
labels = np.load("models/cluster_labels.npy")
kmeans = joblib.load("models/kmeans.pkl")
nb = joblib.load("models/nb_model.pkl")
rules = pd.read_csv("models/rules.csv")

# Test K-Means
print(f"Silhouette Score: {silhouette_score(combined[:1000], labels[:1000]):.3f}")  # >= 0.3
print(f"So cum: {len(set(labels))}")

# Test Naive Bayes (multi-label)
cnn = np.load("models/cnn_features.npy")
mlb = joblib.load("models/mlb_encoder.pkl")
pred = nb.predict(cnn[:5])               # shape (5, n_genres)
genres_pred = mlb.inverse_transform(pred)
print(f"NB predictions: {genres_pred}")

# Test Association Rules
print(f"So luat ket hop: {len(rules)}")
print(rules[['antecedents', 'consequents', 'confidence', 'lift']].head(10))
```

---

## Module 4: Backend API (FastAPI)

### Mục tiêu
Xây dựng REST API server, load tất cả model đã train, phục vụ dữ liệu cho frontend.

### Cấu trúc thư mục
```
backend/
├── app.py              # FastAPI entry point, load models khi startup
├── database.py         # SQLAlchemy models + init DB từ movies.csv
├── recommender.py      # Logic tính cosine similarity + gợi ý
├── routers/
│   ├── movies.py       # /api/movies endpoints
│   ├── recommend.py    # /api/movies/{id}/recommend
│   ├── genres.py       # /api/genres/rules
│   └── clusters.py     # /api/clusters
└── schemas.py          # Pydantic response models
```

### API Endpoints chi tiết

| Method | Endpoint | Response |
|--------|----------|----------|
| GET | `/api/movies?page=1&limit=20` | Danh sách phim phân trang |
| GET | `/api/movies/{id}` | Chi tiết phim + cluster + NB prediction |
| GET | `/api/movies/{id}/recommend?alpha=0.5` | Top-10 phim gợi ý |
| GET | `/api/movies/search?q=inception` | Tìm kiếm theo tên |
| GET | `/api/genres/rules` | Danh sách association rules |
| GET | `/api/clusters` | Thống kê + danh sách phim theo cụm |
| GET | `/api/clusters/{cluster_id}/movies` | Phim trong 1 cụm cụ thể |

### Database Schema (SQLite)
```sql
-- Bảng chính: metadata phim (query nhanh)
CREATE TABLE movies (
    movie_id    INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    year        INTEGER,
    genres      TEXT,           -- JSON array string
    overview    TEXT,
    poster_url  TEXT,           -- URL TMDB trực tiếp
    rating      REAL,
    vote_count  INTEGER,
    cluster_id  INTEGER,        -- K-Means output
    pca_x       REAL,           -- Precomputed PCA coordinate
    pca_y       REAL,           -- Precomputed PCA coordinate
    predicted_genres TEXT       -- MultiOutputClassifier output (JSON)
);

-- Bảng association rules
CREATE TABLE rules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    antecedent  TEXT,
    consequent  TEXT,
    support     REAL,
    confidence  REAL,
    lift        REAL
);
```
> Feature vectors (CNN, TF-IDF, combined) lưu trong `.npy` — không vào DB vì cần numpy để tính similarity

### Startup sequence trong `app.py`
```python
@app.on_event("startup")
async def load_models():
    # Load numpy arrays vào RAM (dùng cho cosine similarity)
    app.state.combined = np.load("models/combined_features.npy")
    app.state.movie_ids = np.load("models/movie_ids.npy")   # mapping index → movie_id
    # Load models
    app.state.nb = joblib.load("models/nb_model.pkl")
    app.state.mlb = joblib.load("models/mlb_encoder.pkl")
    # Metadata query từ SQLite qua SQLAlchemy (không load hết vào RAM)
```

### Cách test Module 4
```bash
# Chạy server
cd backend
uvicorn app:app --reload --port 8000

# Test thủ công qua Swagger UI
# Mở trình duyệt: http://localhost:8000/docs

# Hoặc test bằng curl
curl http://localhost:8000/api/movies?page=1&limit=5
curl http://localhost:8000/api/movies/550/recommend
curl http://localhost:8000/api/genres/rules
```

```python
# test_api.py — automated test
import requests

BASE = "http://localhost:8000"

# Test 1: Danh sach phim
r = requests.get(f"{BASE}/api/movies?limit=5")
assert r.status_code == 200
assert len(r.json()["movies"]) == 5

# Test 2: Chi tiet phim
r = requests.get(f"{BASE}/api/movies/550")
assert "title" in r.json()
assert "cluster_id" in r.json()
assert "predicted_genre" in r.json()

# Test 3: Goi y phim
r = requests.get(f"{BASE}/api/movies/550/recommend")
assert len(r.json()["recommendations"]) == 10

# Test 4: Luat ket hop
r = requests.get(f"{BASE}/api/genres/rules")
assert len(r.json()["rules"]) > 0

print("Tat ca API tests passed!")
```

---

## Module 5: Frontend (React + Vite + Tailwind)

### Mục tiêu
Giao diện web trực quan, hiển thị kết quả từ tất cả các mô hình ML.

### Cấu trúc thư mục
```
frontend/
├── src/
│   ├── api/
│   │   └── movieApi.js         # Axios calls tới backend
│   ├── components/
│   │   ├── MovieCard.jsx        # Card hiển thị poster + tên phim
│   │   ├── MovieGrid.jsx        # Lưới phim
│   │   ├── SearchBar.jsx        # Thanh tìm kiếm
│   │   ├── GenreTag.jsx         # Badge thể loại
│   │   ├── WeightSlider.jsx     # Điều chỉnh alpha poster/text
│   │   ├── ClusterChart.jsx     # PCA 2D scatter plot (recharts)
│   │   └── RulesTable.jsx       # Bảng association rules
│   ├── pages/
│   │   ├── HomePage.jsx         # Trang chủ: search + featured
│   │   ├── MovieDetailPage.jsx  # Chi tiết phim + gợi ý
│   │   └── ExplorePage.jsx      # Visualize cụm + rules
│   ├── App.jsx
│   └── main.jsx
├── index.html
├── vite.config.js
└── tailwind.config.js
```

### Chiến lược phát triển: Mock trước, API sau

```javascript
// src/api/movieApi.js
const USE_MOCK = true;  // Bật mock khi chưa có backend

export const getMovies = async (page = 1) => {
  if (USE_MOCK) return mockMovies;
  return axios.get(`/api/movies?page=${page}`).then(r => r.data);
};
```

### Các trang chính

**Trang Home (`/`):**
- Thanh search realtime
- Grid phim nổi bật (20 phim/trang, phân trang)
- Filter theo thể loại (dựa trên association rules)

**Trang Detail (`/movie/:id`):**
- Banner poster lớn + metadata
- Badge "Thể loại dự đoán bởi AI" (Naive Bayes)
- Slider alpha điều chỉnh trọng số poster vs text
- Grid top-10 phim gợi ý
- Badge cụm phim (K-Means cluster)

**Trang Explore (`/explore`):**
- Scatter plot PCA 2D các cụm (màu sắc theo cluster)
- Bảng association rules (antecedent → consequent, confidence, lift)
- Thống kê số phim/cụm (bar chart)

### Cách test Module 5
```bash
# Chạy dev server
cd frontend
npm run dev
# Mở http://localhost:5173

# Build production
npm run build
npm run preview
```

**Checklist test thủ công:**
- [ ] Search "Inception" → hiển thị đúng phim
- [ ] Click vào phim → mở trang detail
- [ ] Kéo slider alpha → danh sách gợi ý thay đổi
- [ ] Trang Explore → scatter plot render đúng
- [ ] Filter thể loại "Action" → chỉ hiện phim Action
- [ ] Responsive trên mobile (375px)

---

## Thứ Tự Thực Hiện & Dependencies

```
Module 1 (Data)
    ↓ movies.csv + posters/
Module 2 (Features)
    ↓ *.npy files
Module 3 (ML Models)
    ↓ *.pkl files
Module 4 (API)          ←→ Module 5 (Frontend, dùng mock song song)
    ↓ kết nối thật
Module 5 (Frontend hoàn chỉnh)
```

**Có thể làm song song:** Module 5 (với mock data) + Module 4 sau khi Module 3 xong

---

## Checklist Tổng

### Module 1
- [ ] `movies.csv` có >= 4500 dòng, không missing
- [ ] Số file poster = số dòng trong CSV
- [ ] Text đã được clean (lowercase, no stopwords)

### Module 2
- [ ] `combined_features.npy` shape `(N, 2548)`
- [ ] Tất cả values trong [0, 1]
- [ ] Cosine similarity: phim cùng thể loại > 0.7

### Module 3
- [ ] Silhouette Score >= 0.3
- [ ] Naive Bayes Accuracy >= 50%
- [ ] Association Rules: >= 10 luật có lift > 1.0

### Module 4
- [ ] Tất cả 7 endpoints trả về HTTP 200
- [ ] Swagger UI hiển thị đầy đủ tại `/docs`
- [ ] Response time < 2s cho `/recommend`

### Module 5
- [ ] 3 trang render không lỗi
- [ ] Search hoạt động
- [ ] Slider alpha thay đổi kết quả gợi ý
- [ ] Kết nối API thật (không dùng mock)

---

## Ước Lượng Thời Gian

| Module | Thời gian |
|--------|-----------|
| Module 1: Data Pipeline | 2-3 ngày |
| Module 2: Feature Extraction | 1-2 ngày |
| Module 3: ML Models | 2-3 ngày |
| Module 4: Backend API | 2-3 ngày |
| Module 5: Frontend | 3-4 ngày |
| Tích hợp + kiểm thử + báo cáo | 2-3 ngày |
| **Tổng** | **~3 tuần** |
