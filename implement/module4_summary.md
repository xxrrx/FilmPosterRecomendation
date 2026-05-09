# Module 4: Backend API (FastAPI) - Tong Ket Thuc Hien

## Trang Thai
**HOAN THANH**

---

## Muc Tieu
Xay dung REST API server, load tat ca model da train, phuc vu du lieu cho frontend.

---

## Cau Truc Thu Muc

```
backend/
├── app.py           — FastAPI entry point, load models luc startup
├── database.py      — SQLAlchemy ORM + import CSV vao SQLite
├── recommender.py   — Logic cosine similarity voi alpha weighting
├── schemas.py       — Pydantic response models
├── routers/
│   ├── movies.py    — GET /api/movies, /search, /{id}
│   ├── recommend.py — GET /api/movies/{id}/recommend
│   ├── genres.py    — GET /api/genres/rules
│   └── clusters.py  — GET /api/clusters, /{id}/movies
└── test_api.py      — Automated test tat ca endpoints
```

---

## 7 Endpoints Da Implement

| Method | Endpoint | Mo ta |
|--------|----------|-------|
| GET | `/api/movies?page=&limit=&genre=` | Danh sach phim phan trang, loc theo the loai |
| GET | `/api/movies/search?q=` | Tim kiem phim theo ten (case-insensitive) |
| GET | `/api/movies/{id}` | Chi tiet phim + cluster_id + predicted_genres |
| GET | `/api/movies/{id}/recommend?alpha=&top_k=` | Top-k goi y, dieu chinh trong so poster/text |
| GET | `/api/genres/rules?min_confidence=&min_lift=` | Danh sach association rules |
| GET | `/api/clusters` | Thong ke tong quat cac cum |
| GET | `/api/clusters/{id}/movies` | Phim trong 1 cum cu the |

---

## Database Schema (SQLite)

```sql
CREATE TABLE movies (
    movie_id         INTEGER PRIMARY KEY,
    title            TEXT,
    year             INTEGER,
    genres           TEXT,    -- JSON array
    overview         TEXT,
    poster_url       TEXT,
    rating           REAL,
    vote_count       INTEGER,
    cluster_id       INTEGER,
    pca_x            REAL,
    pca_y            REAL,
    predicted_genres TEXT     -- JSON array (Naive Bayes output)
);

CREATE TABLE rules (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    antecedent TEXT,
    consequent TEXT,
    support    REAL,
    confidence REAL,
    lift       REAL
);
```

---

## Logic Quan Trong

### Startup Sequence
1. Load `combined_features.npy`, `movie_ids.npy`, `cluster_labels.npy` vao RAM
2. Load `nb_model.pkl`, `mlb_encoder.pkl`
3. Khoi tao `Recommender` object
4. Goi `init_db()` — import CSV vao SQLite neu lan dau chay

### Recommender Logic
- Lay combined vector cua phim goc
- Ap dung alpha weighting: `[alpha * cnn | (1-alpha) * tfidf]`
- Tim kiem trong cung cum K-Means (thu hep khong gian tim kiem)
- Tinh cosine similarity, tra ve top-k ket qua

### Tinh Nang Khac
- CORS middleware cho phep frontend (localhost:5173) goi API
- Swagger UI tu dong tai `/docs`
- Filter rules theo `min_confidence`, `min_lift`

---

## Ket Qua Chay Server

```
[Startup] combined_features: (4752, 2548)
[Startup] movie_ids: (4752,)
[DB] Da import 4752 phim.
[DB] Da import 12 luat ket hop.
[Startup] San sang! Truy cap http://localhost:8000/docs
```

---

## Cach Chay
```bash
cd backend
uvicorn app:app --reload --port 8000
# Swagger UI: http://127.0.0.1:8000/docs
# Test: python test_api.py
```
