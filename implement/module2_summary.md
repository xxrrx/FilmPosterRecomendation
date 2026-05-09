# Module 2: Feature Extraction - Tong Ket Thuc Hien

## Trang Thai
**HOAN THANH**

---

## Muc Tieu
Trich xuat dac trung tu anh poster (CNN) va van ban mo ta (TF-IDF), ket hop thanh vector tong hop de phuc vu clustering va recommendation.

---

## Nhung Gi Da Thuc Hien

### 2.1 - Load ResNet50 Pretrained
- Su dung `ResNet50(weights='imagenet', include_top=False, pooling='avg')`
- Bo lop FC cuoi, lay Global Average Pooling → output 2048 chieu
- Dong bang model (`trainable=False`), chi dung de extract features

### 2.2 - CNN Feature Extraction tu Poster URL
- Fetch poster tu TMDB URL (`https://image.tmdb.org/t/p/w500/...`)
- Resize ve `224x224 RGB`, ap dung `preprocess_input` cua ResNet50
- Xu ly theo **batch_size=32**, fetch anh song song bang **20 threads**
- Co **checkpoint tu dong** moi 10 batch — resume duoc neu bi ngat
- Ket qua: ma tran `(4752, 2048)`

### 2.3 - TF-IDF Feature Extraction
- Ket hop `overview_clean + genres` thanh 1 chuoi text
- `TfidfVectorizer(max_features=500, ngram_range=(1,2), min_df=2, sublinear_tf=True)`
- Ket qua: ma tran `(4752, 500)`

### 2.4 - Normalize ve [0, 1]
- `MinMaxScaler` rieng biet cho CNN va TF-IDF
- Luu 2 scaler vao `models/scalers.pkl`

### 2.5 - Concatenate → Combined Features
- `combined = [cnn_normalized | tfidf_normalized]`
- Ket qua: `(4752, 2548)` — tat ca gia tri trong `[0.000, 1.000]`

### 2.6 - Luu File Output
- Luu numpy arrays, scalers, vectorizer, movie_ids
- Luu `movies_valid.csv` (subset phim co poster hop le)

---

## Ket Qua Dau Ra

### File Output
| File | Shape / Mo ta |
|------|---------------|
| `models/cnn_features.npy` | (4752, 2048) — raw CNN |
| `models/tfidf_matrix.npy` | (4752, 500) — raw TF-IDF |
| `models/combined_features.npy` | (4752, 2548) — normalized + concat |
| `models/movie_ids.npy` | (4752,) — mapping index → movie_id |
| `models/tfidf_vectorizer.pkl` | De transform phim moi |
| `models/scalers.pkl` | MinMaxScaler CNN + TF-IDF |
| `data/processed/movies_valid.csv` | 4752 phim co poster hop le |

### Thong Ke
| Chi so | Gia tri |
|--------|---------|
| So phim co CNN features | 4752 |
| CNN features shape | (4752, 2048) |
| TF-IDF shape | (4752, 500) |
| Combined shape | (4752, 2548) |
| Combined range | [0.000, 1.000] |

---

## Kiem Tra Dau Ra (Test Pass)
- Shape CNN (N, 2048): **PASS**
- Shape TF-IDF (N, 500): **PASS**
- Shape Combined (N, 2548): **PASS**
- Combined range [0, 1]: **PASS**
- Cosine similarity phim cung the loai > phim khac: **PASS**
