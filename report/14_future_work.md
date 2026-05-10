# Chương 14: Hướng Phát Triển Tương Lai

## 14.1 Nâng Cấp Mô Hình Đặc Trưng

### 14.1.1 Thay Thế TF-IDF bằng Sentence-BERT

**Vấn đề hiện tại:** TF-IDF không hiểu ngữ nghĩa — "alien invasion" và "extraterrestrial attack" có vector hoàn toàn khác nhau dù cùng nghĩa.

**Đề xuất:** Sử dụng **Sentence-BERT** (SBERT) để mã hóa mô tả phim thành vector ngữ nghĩa dày đặc (dense semantic embeddings):

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 chiều
embeddings = model.encode(df['overview_clean'].tolist(),
                          batch_size=64,
                          show_progress_bar=True)
# Shape: (4768, 384)
```

**Lợi ích:**
- Hiểu ngữ nghĩa sâu: cùng nghĩa → gần nhau trong không gian vector
- Không phụ thuộc vào từ vựng cụ thể (không bị ảnh hưởng bởi đồng nghĩa)
- 384 chiều (thay vì 500) — nhỏ hơn nhưng phong phú hơn

**Thách thức:** Cần ~5–10 phút để mã hóa 4,768 phim (CPU); nhưng chỉ cần chạy một lần.

### 14.1.2 Fine-tuning CNN với Poster Phim

**Vấn đề hiện tại:** ResNet50 được huấn luyện trên ImageNet (phân loại đối tượng tự nhiên) — không tối ưu cho đặc trưng thẩm mỹ điện ảnh.

**Đề xuất:** Fine-tune ResNet50 với task phân loại thể loại phim từ poster:

```python
# Thêm classification head
base_model = ResNet50(weights='imagenet', include_top=False)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
output = Dense(17, activation='sigmoid')(x)  # Multi-label sigmoid

model = Model(base_model.input, output)

# Freeze base layers, chỉ train top layers trước
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(poster_images, genre_labels, epochs=10, batch_size=32)
```

**Lợi ích:** Đặc trưng học được sẽ phản ánh thể loại và phong cách điện ảnh tốt hơn đặc trưng ImageNet chung.

### 14.1.3 Vision Transformer (ViT) cho Poster

**Đề xuất:** Thay ResNet50 bằng ViT-B/16 — kiến trúc Transformer cho hình ảnh đạt hiệu suất cao nhất hiện nay:

```python
from transformers import ViTModel, ViTFeatureExtractor

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

inputs = feature_extractor(images=img_list, return_tensors='pt')
outputs = model(**inputs)
# CLS token embedding: (N, 768)
```

ViT có lợi thế xử lý toàn cục (global attention) so với CNN xử lý cục bộ (local receptive field).

---

## 14.2 Hybrid Recommendation System

### 14.2.1 Kết Hợp Collaborative Filtering

**Đề xuất:** Thu thập dữ liệu người dùng và xây dựng Hybrid Recommender:

```
score_hybrid(u, m) = β × CF_score(u, m) + (1-β) × CB_score(m_q, m)
```

trong đó:
- `CF_score(u, m)` từ Matrix Factorization (ALS, SVD++)
- `CB_score(m_q, m)` từ cosine similarity hiện tại (KhaiPha)
- `β` là tham số blending

**Kế hoạch thu thập dữ liệu:**
1. Thêm hệ thống đăng nhập người dùng vào frontend
2. Log các sự kiện: click, watch time, explicit rating (1-5 stars)
3. Sau khi đủ dữ liệu (~1000 users × 50 interactions), huấn luyện CF

### 14.2.2 Learning-to-Rank

Thay vì blending cố định với alpha, huấn luyện mô hình Learning-to-Rank:

```python
from lightgbm import LGBMRanker

ranker = LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    n_estimators=100
)
# Features: [cnn_sim, tfidf_sim, rating_diff, year_diff, genre_overlap]
# Labels: relevance scores từ user feedback
ranker.fit(X_features, y_relevance, group=query_groups)
```

---

## 14.3 Cải Thiện Phân Cụm

### 14.3.1 UMAP + HDBSCAN

**Vấn đề:** K-Means giả định cụm hình cầu và cần K cố định.

**Đề xuất:**
1. **UMAP** (Uniform Manifold Approximation and Projection): Giảm chiều về 50D (thay vì 2D) giữ được cấu trúc cục bộ tốt hơn PCA:

```python
import umap

reducer = umap.UMAP(n_components=50, n_neighbors=15, min_dist=0.1)
features_umap = reducer.fit_transform(combined_features)  # (4768, 50)
```

2. **HDBSCAN**: Phân cụm density-based, tự động xác định K và hình dạng cụm tùy ý:

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5)
cluster_labels = clusterer.fit_predict(features_umap)
```

**Lợi ích:** Silhouette Score dự kiến cải thiện đáng kể từ -0.003 lên >0.2.

### 14.3.2 Autoencoders cho Biểu Diễn Nén

```python
from tensorflow.keras import layers, Model

# Encoder
inputs = layers.Input(shape=(2548,))
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(128, activation='relu')(x)
encoded = layers.Dense(64, activation='relu', name='bottleneck')(x)

# Decoder
x = layers.Dense(128, activation='relu')(encoded)
x = layers.Dense(512, activation='relu')(x)
decoded = layers.Dense(2548, activation='sigmoid')(x)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(combined_features, combined_features, epochs=50, batch_size=64)

# Bottleneck features (64 chiều)
encoder = Model(inputs, encoded)
compressed = encoder.predict(combined_features)  # (4768, 64)
```

Biểu diễn 64 chiều từ bottleneck loại bỏ nhiễu và giữ lại thông tin cần thiết.

---

## 14.4 Nâng Cấp Hạ Tầng

### 14.4.1 FAISS cho Similarity Search Nhanh

Facebook AI Similarity Search (FAISS) cho phép tìm kiếm approximate nearest neighbors cực nhanh:

```python
import faiss

# Xây dựng index
d = combined_features.shape[1]  # 2548
index = faiss.IndexFlatIP(d)    # Inner Product (cosine sau khi L2 normalize)
index.add(combined_features.astype('float32'))

# Query: k nearest neighbors
D, I = index.search(query_vector.reshape(1, -1), k=10)
# D: distances (4768,), I: indices (10,)
```

FAISS giảm thời gian query từ ~80ms xuống ~5ms với 4,768 phim; hiệu quả hơn nhiều với triệu phim.

### 14.4.2 Chuyển sang PostgreSQL + pgvector

Khi scale lên hàng triệu phim, SQLite không còn phù hợp:

```sql
-- PostgreSQL với pgvector extension
CREATE EXTENSION vector;

CREATE TABLE movies (
    id SERIAL PRIMARY KEY,
    movie_id INTEGER UNIQUE,
    title VARCHAR(255),
    combined_embedding vector(2548)  -- pgvector type
);

-- Approximate nearest neighbor search
SELECT title, 1 - (combined_embedding <=> query_embedding) AS similarity
FROM movies
ORDER BY combined_embedding <=> query_embedding
LIMIT 10;
```

### 14.4.3 Caching với Redis

```python
import redis
import json

cache = redis.Redis(host='localhost', port=6379)

def get_recommendations_cached(movie_id, alpha):
    cache_key = f"rec:{movie_id}:{alpha}"
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)

    results = compute_recommendations(movie_id, alpha)
    cache.setex(cache_key, 3600, json.dumps(results))  # TTL 1 giờ
    return results
```

---

## 14.5 Cập Nhật Dữ Liệu Tự Động

### 14.5.1 Pipeline ETL Tự Động

```python
# scheduled_update.py (chạy hàng tuần)
import schedule

def update_movie_database():
    # 1. Fetch phim mới từ TMDB API (released trong 7 ngày qua)
    new_movies = tmdb_api.get_recently_released(days=7)
    # 2. Fetch poster, clean text
    processed = preprocess_new_movies(new_movies)
    # 3. Extract CNN + TF-IDF features
    new_features = extract_features(processed)
    # 4. Append vào combined_features matrix
    update_feature_matrix(new_features)
    # 5. Rebuild FAISS index
    rebuild_index()

schedule.every().week.do(update_movie_database)
```

---

## 14.6 Cải Thiện Giao Diện Người Dùng

### 14.6.1 Giải Thích Gợi Ý (Explainable AI)

Hiển thị lý do tại sao hai phim được coi là tương đồng:

```
"Interstellar được gợi ý cho Avatar vì:
  - Poster: Cả hai có tông màu lạnh và cảnh không gian (CNN similarity: 0.53)
  - Nội dung: Cả hai về khám phá không gian và nhân tính (TF-IDF similarity: 0.51)
  - Thể loại: Cả hai thuộc Action + Sci-Fi"
```

### 14.6.2 Giao Diện Đa Ngôn Ngữ

Hỗ trợ tìm kiếm và hiển thị tiếng Việt (dịch metadata từ TMDB API multilingual support).

---

## 14.7 Tổng Kết Lộ Trình Phát Triển

| Giai đoạn | Thời gian | Ưu tiên | Nội dung |
|-----------|-----------|---------|---------|
| Ngắn hạn | 1–2 tháng | Cao | Sentence-BERT, FAISS, Redis cache |
| Trung hạn | 3–6 tháng | Trung bình | User feedback system, UMAP+HDBSCAN, pgvector |
| Dài hạn | 6–12 tháng | Thấp | ViT fine-tuned, Hybrid CF, explainable AI |
| Nghiên cứu | Ongoing | — | Learning-to-rank, Autoencoder, multimodal fusion |
