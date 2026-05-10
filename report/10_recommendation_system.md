# Chương 10: Hệ Thống Gợi Ý Tổng Thể

## 10.1 Kiến Trúc Tổng Quan

Hệ thống gợi ý KhaiPha được thiết kế theo kiến trúc 3 tầng (three-tier architecture):

```
[Tầng 1: Data Layer]
    SQLite Database (movies, clusters)
    NumPy Arrays (cnn_features, tfidf_matrix, combined_features)
    Pickle Files (kmeans, nb_model, tfidf_vectorizer)

[Tầng 2: Application Layer]
    FastAPI Backend
    Cosine Similarity Engine (recommender.py)
    Naive Bayes Inference
    Association Rules Loader

[Tầng 3: Presentation Layer]
    React 18 Frontend
    Vite Build System
    Tailwind CSS
    Recharts Visualization
```

---

## 10.2 Pipeline Gợi Ý

### 10.2.1 Luồng Xử Lý Tổng Thể

Khi người dùng yêu cầu gợi ý cho phim `m_q`:

```
Bước 1: Nhận request → GET /api/movies/{movie_id}/recommend?alpha=0.5
Bước 2: Tra cứu movie_id → vị trí hàng trong ma trận đặc trưng
Bước 3: Lấy vector CNN: cnn_q = cnn_features[row_idx]   (2048 chiều)
Bước 4: Lấy vector TF-IDF: tfidf_q = tfidf_matrix[row_idx] (500 chiều)
Bước 5: Tính cosine similarity với tất cả phim
Bước 6: Blend CNN + TF-IDF theo alpha
Bước 7: Sắp xếp giảm dần, loại m_q, lấy top-10
Bước 8: Truy vấn metadata từ SQLite
Bước 9: Trả về JSON response
```

### 10.2.2 Chi Tiết Tính Toán Similarity

```python
# backend/recommender.py

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Recommender:
    def __init__(self):
        self.cnn_features = np.load('models/cnn_features.npy')      # (4768, 2048)
        self.tfidf_matrix = np.load('models/tfidf_matrix.npy')      # (4768, 500)
        self.combined = np.load('models/combined_features.npy')      # (4768, 2548)
        self.movie_ids = np.load('models/movie_ids.npy')             # (4768,)

    def get_recommendations(self, movie_id: int, alpha: float = 0.5, k: int = 10):
        # Tìm index của phim truy vấn
        idx = np.where(self.movie_ids == movie_id)[0]
        if len(idx) == 0:
            return []
        idx = idx[0]

        # Tính cosine similarity riêng biệt cho CNN và TF-IDF
        cnn_sim = cosine_similarity(
            [self.cnn_features[idx]],
            self.cnn_features
        )[0]  # shape: (4768,)

        tfidf_sim = cosine_similarity(
            [self.tfidf_matrix[idx]],
            self.tfidf_matrix
        )[0]  # shape: (4768,)

        # Blending theo alpha
        blended_sim = alpha * cnn_sim + (1 - alpha) * tfidf_sim

        # Sắp xếp và lọc
        sorted_indices = np.argsort(blended_sim)[::-1]
        # Loại bỏ chính phim truy vấn
        sorted_indices = sorted_indices[sorted_indices != idx]
        top_k_indices = sorted_indices[:k]

        # Lấy movie_ids và similarity scores
        result_ids = self.movie_ids[top_k_indices]
        result_scores = blended_sim[top_k_indices]

        return list(zip(result_ids.tolist(), result_scores.tolist()))
```

### 10.2.3 Công Thức Blending

```
sim_final(m_q, m_i) = alpha × cos_sim(CNN(m_q), CNN(m_i))
                    + (1 - alpha) × cos_sim(TFIDF(m_q), TFIDF(m_i))
```

**Ví dụ với Avatar (movie_id=19995, alpha=0.5):**

| Hạng | Phim | CNN_sim | TFIDF_sim | Blended (α=0.5) |
|------|------|---------|-----------|-----------------|
| 1 | Total Recall | 0.619 | 0.421 | 0.520 |
| 2 | Iron Man | 0.566 | 0.398 | 0.482 |
| 3 | Star Trek | 0.541 | 0.439 | 0.490 |
| 4 | Interstellar | 0.529 | 0.512 | 0.521 |
| 5 | Guardians | 0.551 | 0.376 | 0.464 |

---

## 10.3 Backend FastAPI

### 10.3.1 Kiến Trúc Ứng Dụng

```
backend/
├── app.py              # Khởi tạo FastAPI, mount routers, startup events
├── database.py         # SQLAlchemy engine, Session, Movie model
├── recommender.py      # Recommender class (in-memory numpy ops)
├── schemas.py          # Pydantic response models
└── routers/
    ├── movies.py       # /api/movies endpoints
    ├── recommend.py    # /api/movies/{id}/recommend
    ├── genres.py       # /api/genres/rules
    └── clusters.py     # /api/clusters endpoints
```

### 10.3.2 Startup Event

```python
# backend/app.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

recommender = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    recommender = Recommender()      # Nạp toàn bộ numpy arrays vào RAM
    load_database()                  # Nạp movies.csv vào SQLite
    yield
    # Cleanup khi shutdown

app = FastAPI(lifespan=lifespan, title="KhaiPha Movie API")
```

Tất cả ma trận đặc trưng (~46 MB) được nạp vào RAM một lần khi khởi động, đảm bảo response time thấp cho mỗi request.

### 10.3.3 Danh Sách Endpoints

| Method | Endpoint | Mô tả | Response |
|--------|----------|-------|---------|
| GET | `/api/movies` | Danh sách phim phân trang | `{movies: [...], total, page, limit}` |
| GET | `/api/movies?q=avatar` | Tìm kiếm theo tên | Danh sách kết quả |
| GET | `/api/movies/{id}` | Chi tiết phim + dự đoán thể loại NB | Movie object + predicted_genres |
| GET | `/api/movies/{id}/recommend` | Top-10 gợi ý | `[{movie, similarity_pct}]` |
| GET | `/api/genres/rules` | 12 luật kết hợp | `[{antecedents, consequents, ...}]` |
| GET | `/api/clusters` | Thống kê 20 cụm | `[{cluster_id, size, avg_rating}]` |
| GET | `/api/clusters/{id}/movies` | Phim trong cụm | Danh sách phim |
| GET | `/docs` | Swagger UI tự động | Giao diện test API |

### 10.3.4 Database Schema

```python
# backend/database.py
from sqlalchemy import Column, Integer, Float, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Movie(Base):
    __tablename__ = 'movies'

    id = Column(Integer, primary_key=True)
    movie_id = Column(Integer, unique=True, index=True)
    title = Column(String(255), index=True)
    year = Column(Integer)
    genres = Column(Text)           # JSON string
    overview_clean = Column(Text)
    poster_url = Column(Text)
    rating = Column(Float)
    vote_count = Column(Integer)
    cluster_id = Column(Integer, index=True)
    pca_x = Column(Float)
    pca_y = Column(Float)
```

SQLite được chọn vì:
- Không cần cấu hình server riêng (zero-config)
- Đủ hiệu năng cho dataset 4,768 phim
- Dễ deploy (single file database)

### 10.3.5 Pydantic Response Models

```python
# backend/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class MovieBase(BaseModel):
    movie_id: int
    title: str
    year: int
    genres: str
    poster_url: Optional[str]
    rating: float
    vote_count: int
    cluster_id: int

class MovieDetail(MovieBase):
    overview_clean: str
    pca_x: float
    pca_y: float
    predicted_genres: List[str]  # Từ Naive Bayes

class RecommendationItem(BaseModel):
    movie: MovieBase
    similarity_pct: float        # 0–100%
```

---

## 10.4 Frontend React

### 10.4.1 Cấu Trúc

```
frontend/src/
├── api/
│   └── movieApi.js          # Axios client với base URL cố định
├── components/
│   ├── MovieCard.jsx         # Thumbnail phim (poster, title, rating)
│   ├── MovieGrid.jsx         # Grid layout responsive 2–5 cột
│   ├── GenreFilter.jsx       # Filter dropdown 17 thể loại
│   ├── SearchBar.jsx         # Ô tìm kiếm với debounce 300ms
│   ├── AlphaSlider.jsx       # Slider điều chỉnh alpha (0.0–1.0)
│   ├── ClusterChart.jsx      # Recharts BarChart kích thước cụm
│   └── PCAScatter.jsx        # Recharts ScatterChart tương tác
└── pages/
    ├── Home.jsx              # Trang chủ: tìm kiếm + lọc + danh sách
    ├── MovieDetail.jsx       # Chi tiết phim + gợi ý
    └── Explore.jsx           # Khám phá: PCA scatter + association rules
```

### 10.4.2 Trang Home

Trang chủ cung cấp:
- **SearchBar với debounce:** Người dùng nhập → chờ 300ms → gọi `GET /api/movies?q={query}`
- **Genre Filter:** Dropdown chọn thể loại → gọi API với tham số `?genre=Action`
- **Pagination:** Phân trang 50 phim/trang
- **Responsive Grid:** 2 cột mobile, 3 cột tablet, 5 cột desktop (Tailwind breakpoints)

### 10.4.3 Trang Movie Detail

Trang chi tiết phim là điểm cốt lõi của trải nghiệm người dùng:

```jsx
// pages/MovieDetail.jsx (simplified)
const MovieDetail = () => {
    const { id } = useParams()
    const [alpha, setAlpha] = useState(0.5)
    const [recommendations, setRecommendations] = useState([])

    useEffect(() => {
        fetchRecommendations(id, alpha)
    }, [id, alpha])  // Gọi lại khi alpha thay đổi

    return (
        <div>
            {/* Poster + Metadata */}
            <MovieHeader movie={movie} />

            {/* Thể loại gốc vs NB predictions */}
            <GenreBadges actual={movie.genres} predicted={movie.predicted_genres} />

            {/* Alpha Slider */}
            <AlphaSlider value={alpha} onChange={setAlpha} />
            <p>alpha={alpha.toFixed(1)} → {alpha===0 ? 'Text only' : alpha===1 ? 'Image only' : 'Balanced'}</p>

            {/* Top 10 Recommendations */}
            <MovieGrid movies={recommendations} />
        </div>
    )
}
```

**AlphaSlider:** Khi người dùng kéo slider, `useEffect` tự động gọi lại API với alpha mới và cập nhật danh sách gợi ý. Trải nghiệm này cho phép người dùng **trực tiếp quan sát** sự thay đổi gợi ý khi trọng số hình ảnh/văn bản thay đổi — tạo ra giá trị giáo dục về cách hệ thống hoạt động.

### 10.4.4 Trang Explore

Trang khám phá cung cấp 3 chức năng:

**1. PCA Scatter Plot tương tác (Recharts ScatterChart):**
```jsx
<ScatterChart>
    <Scatter data={allMovies}
             dataKey={{ x: 'pca_x', y: 'pca_y' }}
             fill={clusterColorMap[point.cluster_id]}
             onClick={(point) => navigate(`/movies/${point.movie_id}`)} />
</ScatterChart>
```
Click vào điểm bất kỳ → chuyển đến trang chi tiết phim đó.

**2. Cluster Size Bar Chart:** Hiển thị số phim trong mỗi cụm 0–19.

**3. Association Rules Table:**
- Bảng hiển thị 12 luật kết hợp với support, confidence, lift
- Có thể lọc theo thể loại
- Sắp xếp theo confidence hoặc lift

---

## 10.5 API Client

```javascript
// frontend/src/api/movieApi.js
import axios from 'axios'

const api = axios.create({
    baseURL: 'http://localhost:8000/api',
    timeout: 10000,
})

export const getMovies = (page=1, limit=50, q='', genre='') =>
    api.get('/movies', { params: {page, limit, q, genre} })

export const getMovieDetail = (id) =>
    api.get(`/movies/${id}`)

export const getRecommendations = (id, alpha=0.5) =>
    api.get(`/movies/${id}/recommend`, { params: {alpha} })

export const getAssociationRules = () =>
    api.get('/genres/rules')

export const getClusters = () =>
    api.get('/clusters')
```

---

## 10.6 Hiệu Năng Hệ Thống

### 10.6.1 Thời Gian Response

| Endpoint | Thời gian TB | Ghi chú |
|----------|-------------|---------|
| `GET /api/movies` | ~15ms | SQLite query + serialization |
| `GET /api/movies/{id}` | ~20ms | SQLite + NB inference (fast) |
| `GET /api/movies/{id}/recommend` | ~80ms | Cosine similarity 4,768 × 2,548 |
| `GET /api/genres/rules` | ~5ms | Load từ file CSV |
| `GET /api/clusters` | ~10ms | SQLite aggregation |

**Bottleneck:** Tính cosine similarity trên 4,768 phim × 2,548 chiều (~80ms). Với scikit-learn's vectorized NumPy operations, đây là thời gian chấp nhận được. Có thể tối ưu thêm bằng FAISS (Facebook AI Similarity Search) nếu cần scale lên hàng triệu phim.

### 10.6.2 Bộ Nhớ

| Artifact | Kích thước RAM |
|----------|--------------|
| CNN features | 37.1 MB |
| TF-IDF matrix | 9.1 MB |
| Combined features | 46.2 MB |
| SQLite (in-memory cache) | ~5 MB |
| Tổng | **~97 MB** |

Tổng bộ nhớ runtime dưới 200MB — phù hợp với cả máy tính cá nhân thông thường.
