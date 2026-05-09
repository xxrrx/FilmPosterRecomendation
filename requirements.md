# Yêu Cầu Dự Án: Website Gợi Ý Phim Dựa Trên Áp Phích và Nội Dung

## 1. Tổng Quan Dự Án

**Tên dự án:** Hệ thống gợi ý phim đa phương tiện (Multimodal Movie Recommendation System)

**Lĩnh vực:** Giải trí - Nền tảng xem phim trực tuyến

**Mục tiêu:** Xây dựng website gợi ý phim thông minh kết hợp phân tích hình ảnh áp phích (poster) và nội dung văn bản (mô tả phim) để đưa ra gợi ý cá nhân hóa cho người dùng.

---

## 2. Bộ Dữ Liệu

### Nguồn dữ liệu chính
- **TMDB 5000 Movie Dataset** (Kaggle): metadata phim, thể loại, overview, poster_path
- **TMDB API**: tải ảnh poster thực tế (224x224 RGB)
- **MovieLens 25M** (GroupLens): dữ liệu đánh giá và tags từ người dùng

### Yêu cầu dữ liệu
- Tối thiểu 5.000 phim có đầy đủ: tiêu đề, thể loại, mô tả, áp phích
- Ảnh poster: định dạng JPG/PNG, kích thước chuẩn hóa 224x224 pixel
- Metadata: năm phát hành, điểm đánh giá, ngôn ngữ gốc, đạo diễn, diễn viên chính

### Tiền xử lý dữ liệu
- Loại bỏ phim thiếu poster hoặc mô tả
- Chuẩn hóa text: chuyển thường, loại stopwords, lemmatization
- Ảnh poster: **không tải về local**, dùng URL trực tiếp từ TMDB (`https://image.tmdb.org/t/p/w500/{poster_path}`)
- Lưu feature vectors ra file `.npy` để tái sử dụng

---

## 3. Các Thành Phần Kỹ Thuật

### 3.1 Trích Xuất Đặc Trưng (Feature Extraction)

**Đặc trưng hình ảnh (Poster Features):**
- Sử dụng mạng CNN pretrained (ResNet50 hoặc VGG16, pretrained trên ImageNet)
- Bỏ lớp fully-connected cuối, lấy feature vector 2048 chiều
- Áp dụng Global Average Pooling để giảm chiều

**Đặc trưng văn bản (Text Features):**
- TF-IDF Vectorizer trên trường `overview` + `genres` + `tags`
- Tối đa 500 features
- Kết hợp thêm metadata: năm phát hành, ngôn ngữ, điểm trung bình

**Kết hợp đặc trưng:**
- Normalize từng loại về [0, 1] trước khi ghép
- Concatenate: `combined_vector = [poster_feat | tfidf_feat]`

### 3.2 Phân Cụm K-Means

- Input: combined feature vector của tất cả phim
- Số cụm: K = 20 (điều chỉnh dựa trên Elbow Method)
- Mục đích: nhóm phim tương đồng về cả nội dung hình ảnh và văn bản
- Đánh giá: Silhouette Score, Davies-Bouldin Index
- Output: nhãn cụm cho mỗi phim, lưu vào database

### 3.3 Phân Loại Naive Bayes

- Input: đặc trưng CNN từ ảnh poster
- Output: dự đoán **nhiều thể loại** phim (multi-label classification)
- Mô hình: `MultiOutputClassifier(GaussianNB())` — mỗi thể loại là 1 binary classifier độc lập
- Đánh giá: Hamming Loss, Precision, Recall, F1-score (micro/macro)
- Mục đích: kiểm tra khả năng nhận diện thể loại qua hình ảnh poster

### 3.4 Luật Kết Hợp (Association Rules)

- Xây dựng transaction matrix: mỗi phim là 1 transaction, các thể loại là items
- Thuật toán: Apriori
- Ngưỡng: min_support = 0.1, min_confidence = 0.5
- Mục đích: khám phá các cặp/nhóm thể loại thường xuất hiện cùng nhau
- Ứng dụng: hiển thị "Bạn thích Action? Thường đi kèm Adventure và Sci-Fi"

### 3.5 Hệ Thống Gợi Ý Đa Phương Tiện

**Luồng gợi ý:**
1. Người dùng chọn 1 phim làm phim gốc (seed movie)
2. Lấy combined feature vector của phim đó
3. Tìm cụm K-Means chứa phim đó
4. Tính cosine similarity trong cụm
5. Trả về top-10 phim có similarity cao nhất

**Công thức similarity:**
```
similarity(A, B) = cosine(combined_vector_A, combined_vector_B)
                 = (A · B) / (||A|| × ||B||)
```

**Tùy chỉnh trọng số:**
- `alpha`: trọng số cho đặc trưng poster (mặc định 0.5)
- `beta`: trọng số cho đặc trưng text (mặc định 0.5)
- Có thể điều chỉnh qua giao diện

---

## 4. Tech Stack

### Môi Trường
- **GPU:** có (dùng cho CNN feature extraction, giảm thời gian từ ~2h xuống ~5 phút)
- **RAM:** 16GB (đủ để load combined_features ~50MB + SQLite vào memory)
- **Python:** 3.9+

### Backend
- **Framework:** FastAPI (tự sinh API docs `/docs`, hỗ trợ async)
- **Database:** SQLite + SQLAlchemy — lưu metadata phim (id, title, genres, poster_url, cluster_id, pca_x, pca_y, predicted_genres); features lưu riêng trong `.npy`
- **Poster:** Dùng URL TMDB trực tiếp (`https://image.tmdb.org/t/p/w500/{poster_path}`), không tải về local
- **Thư viện ML/DL:**

| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| `tensorflow` / `keras` | 2.x | CNN trích đặc trưng poster (ResNet50) |
| `scikit-learn` | latest | K-Means, Naive Bayes, TF-IDF, metrics |
| `mlxtend` | latest | Apriori + Association Rules |
| `numpy` | latest | Xử lý ma trận, feature vectors |
| `pandas` | latest | Xử lý dữ liệu tabular |
| `Pillow` | latest | Đọc/resize ảnh poster cho CNN (fetch từ URL TMDB) |
| `uvicorn` | latest | ASGI server chạy FastAPI |
| `sqlalchemy` | latest | ORM cho SQLite |

### Frontend
- **Framework:** React 18 (Vite) — component-based, dễ quản lý state
- **Styling:** Tailwind CSS — styling nhanh, không cần viết CSS thủ công
- **HTTP Client:** Axios — gọi API backend
- Hiển thị poster phim, thông tin chi tiết, danh sách gợi ý

### Công Cụ Hỗ Trợ
| Công cụ | Mục đích |
|---------|----------|
| Jupyter Notebook | Thực nghiệm, train model, EDA |
| TMDB API | Lấy poster + metadata phim |
| Kaggle CLI | Tải dataset TMDB 5000 |

### Luồng Hoạt Động
```
Jupyter Notebooks (train + export model .pkl / .npy)
        ↓
FastAPI Backend (load model, serve REST API)
        ↓
React Frontend (gọi API, hiển thị gợi ý)
```

## 5. Yêu Cầu Hệ Thống (API)

### API Endpoints (REST)
| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/movies` | Danh sách phim (phân trang) |
| GET | `/api/movies/{id}` | Chi tiết 1 phim |
| GET | `/api/movies/{id}/recommend` | Top-10 phim gợi ý |
| GET | `/api/movies/search?q=` | Tìm kiếm phim theo tên |
| GET | `/api/genres/rules` | Luật kết hợp thể loại |
| GET | `/api/clusters` | Thống kê các cụm phim |

---

## 6. Yêu Cầu Giao Diện

### Trang chủ
- Thanh tìm kiếm phim theo tên
- Hiển thị phim nổi bật theo cụm
- Bộ lọc theo thể loại

### Trang chi tiết phim
- Hiển thị poster, tiêu đề, năm, thể loại, mô tả
- Kết quả phân loại Naive Bayes (thể loại dự đoán từ poster)
- Danh sách top-10 phim gợi ý (hiển thị poster + tên)
- Thanh điều chỉnh trọng số poster/text cho gợi ý

### Trang khám phá
- Scatter plot PCA 2D (tọa độ precomputed, lưu sẵn trong DB — không tính runtime)
- Bảng luật kết hợp thể loại (antecedent → consequent, confidence, lift)
- Thống kê số phim theo từng cụm (bar chart)

---

## 7. Yêu Cầu Phi Chức Năng

| Tiêu chí | Yêu cầu |
|----------|---------|
| Thời gian phản hồi gợi ý | < 2 giây |
| Thời gian tải trang | < 3 giây |
| Số phim tối thiểu trong database | 5.000 phim |
| Độ chính xác Naive Bayes (top-1) | >= 50% |
| Silhouette Score K-Means | >= 0.3 |

---

## 8. Cấu Trúc Thư Mục Dự Án

```
movie-recommender/
├── data/
│   ├── raw/                  # Dữ liệu thô từ TMDB/MovieLens
│   └── processed/            # Dữ liệu sau tiền xử lý (movies.csv)
│                             # Poster dùng URL TMDB, không lưu local
├── models/
│   ├── cnn_features.npy        # shape (N, 2048) — GPU extracted
│   ├── tfidf_matrix.npy        # shape (N, 500)
│   ├── combined_features.npy   # shape (N, 2548) — normalized + concat
│   ├── cluster_labels.npy      # shape (N,) — K-Means output
│   ├── kmeans_model.pkl        # K-Means model
│   ├── nb_model.pkl            # MultiOutputClassifier(GaussianNB)
│   ├── mlb_encoder.pkl         # MultiLabelBinarizer
│   ├── tfidf_vectorizer.pkl    # để transform phim mới
│   ├── scalers.pkl             # MinMaxScaler
│   └── rules.csv               # Association rules output
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_classification.ipynb
│   ├── 05_association_rules.ipynb
│   └── 06_recommendation.ipynb
├── backend/
│   ├── app.py                # FastAPI app entry point
│   ├── recommender.py        # Logic gợi ý
│   ├── database.py           # SQLAlchemy models
│   └── routers/              # API route handlers
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Trang chính
│   │   └── api/              # Axios calls
│   └── public/
├── requirements.txt
└── README.md
```

---

## 9. Kế Hoạch Thực Hiện

| Tuần | Công việc | Deliverable |
|------|-----------|-------------|
| 1 | Thu thập dữ liệu, tải poster, EDA | Notebook EDA + dataset sạch |
| 2 | CNN feature extraction + TF-IDF | File `.npy` chứa features |
| 3 | K-Means + Naive Bayes + Association Rules | 3 mô hình đã train + đánh giá |
| 4 | Recommendation Engine + FastAPI | API hoạt động |
| 5 | Frontend + tích hợp + kiểm thử + báo cáo | Website hoàn chỉnh |

---

## 10. Tiêu Chí Đánh Giá

- **Tính đầy đủ:** Có đủ 4 thành phần (K-Means, Naive Bayes, Association Rules, Recommender)
- **Chất lượng mô hình:** Đạt ngưỡng metrics đề ra
- **Tích hợp đa phương tiện:** Kết hợp thực sự giữa poster và text, không chỉ dùng 1 loại
- **Giao diện:** Trực quan, dễ sử dụng, hiển thị được kết quả từ tất cả các mô hình
- **Báo cáo:** Giải thích rõ phương pháp, kết quả, nhận xét ưu/nhược điểm
