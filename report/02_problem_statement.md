# Chương 2: Phát Biểu Bài Toán

## 2.1 Bài Toán Gợi Ý Phim

### 2.1.1 Định Nghĩa Hình Thức

Cho tập phim `M = {m_1, m_2, ..., m_N}` với N = 4,768 phim. Mỗi phim `m_i` được mô tả bởi:
- **Ảnh poster:** `I_i ∈ R^(224×224×3)` — ảnh RGB kích thước 224×224 pixel.
- **Văn bản mô tả:** `t_i ∈ Σ*` — chuỗi ký tự gồm phần tóm tắt nội dung và thể loại.
- **Metadata:** năm phát hành, điểm đánh giá, số lượt bầu chọn, danh sách thể loại.

**Bài toán gợi ý nội dung (Content-Based Recommendation):**

Cho một phim truy vấn `m_q ∈ M`, tìm tập `R ⊂ M \ {m_q}` gồm K phim tương đồng nhất, sao cho:

```
R = argmax_{m_i ∈ M\{m_q}, |R|=K} sim(m_q, m_i)
```

trong đó `sim(·, ·)` là hàm đo độ tương đồng giữa hai phim trong không gian đặc trưng kết hợp.

### 2.1.2 Không Gian Đặc Trưng

Mỗi phim được biểu diễn trong không gian đặc trưng tích hợp:

```
f_i = [normalize(CNN(I_i)) || normalize(TFIDF(t_i))]
    ∈ R^(2048 + 500)
    = R^2548
```

trong đó:
- `CNN(I_i)` — vector đặc trưng 2,048 chiều từ lớp GAP của ResNet50.
- `TFIDF(t_i)` — vector 500 chiều từ TF-IDF Vectorizer.
- `normalize(·)` — chuẩn hóa MinMax về khoảng [0, 1].
- `||` — phép nối vector (concatenation).

### 2.1.3 Hàm Tương Đồng

Hệ thống sử dụng **độ tương đồng cosine** (cosine similarity) làm hàm đo độ tương đồng giữa hai phim:

```
cos_sim(f_i, f_j) = (f_i · f_j) / (||f_i|| · ||f_j||)
                  ∈ [-1, 1]
```

Trong thực tế, với đặc trưng đã chuẩn hóa MinMax, tất cả giá trị đều không âm, do đó:

```
cos_sim(f_i, f_j) ∈ [0, 1]
```

Giá trị bằng 1 biểu thị hai phim hoàn toàn giống nhau về hướng vector đặc trưng; giá trị 0 biểu thị hai phim hoàn toàn không liên quan.

### 2.1.4 Tham Số Alpha — Blending Đa Phương Thức

Hệ thống cung cấp khả năng điều chỉnh trọng số giữa hai phương thức thông qua tham số alpha:

```
sim_alpha(m_q, m_i) = alpha × cos_sim(CNN(m_q), CNN(m_i))
                    + (1 - alpha) × cos_sim(TFIDF(m_q), TFIDF(m_i))

với alpha ∈ [0.0, 1.0]
```

| Giá trị alpha | Ý nghĩa |
|---------------|---------|
| 0.0 | Chỉ dùng đặc trưng văn bản TF-IDF |
| 0.5 | Kết hợp cân bằng (mặc định) |
| 1.0 | Chỉ dùng đặc trưng hình ảnh CNN |

---

## 2.2 Các Bài Toán Con

Hệ thống KhaiPha giải quyết đồng thời nhiều bài toán con trong Data Mining và Machine Learning:

### 2.2.1 Bài Toán Phân Cụm (Clustering)

Cho tập đặc trưng `F = {f_1, f_2, ..., f_N}`, tìm phân hoạch `C = {C_1, C_2, ..., C_K}` sao cho:

```
argmin_{C} Σ_{k=1}^{K} Σ_{f_i ∈ C_k} ||f_i - μ_k||^2
```

trong đó `μ_k` là tâm cụm thứ k, và K = 20 (chọn qua Elbow Method).

**Mục đích:** Nhóm các phim có nội dung tương đồng để hỗ trợ chức năng duyệt phim theo cụm trong giao diện.

### 2.2.2 Bài Toán Phân Loại Đa Nhãn (Multi-label Classification)

Cho vector đặc trưng CNN `c_i ∈ R^2048`, dự đoán tập nhãn thể loại:

```
ŷ_i = {g ∈ G : P(g | c_i) > θ}
```

trong đó:
- `G = {g_1, ..., g_17}` — tập 17 thể loại phim.
- `θ` — ngưỡng phân loại (mặc định 0.5).
- Mỗi phim có thể thuộc nhiều thể loại đồng thời (multi-label).

### 2.2.3 Bài Toán Khai Phá Luật Kết Hợp (Association Rule Mining)

Cho cơ sở dữ liệu giao dịch thể loại `T = {T_1, T_2, ..., T_N}`, với mỗi `T_i ⊆ G` là tập thể loại của phim thứ i, tìm tất cả luật dạng:

```
X → Y   (X, Y ⊆ G, X ∩ Y = ∅)
```

thỏa mãn:
- `support(X → Y) ≥ min_sup = 0.05`
- `confidence(X → Y) ≥ min_conf = 0.40`
- `lift(X → Y) > 1.0`

---

## 2.3 Phương Pháp Tiếp Cận

Hệ thống KhaiPha áp dụng phương pháp **Content-Based Filtering kết hợp đa phương thức**. Sự lựa chọn này xuất phát từ hai lý do chính:

**Lý do 1 — Không có dữ liệu người dùng:**
Tập dữ liệu TMDB 5000 không chứa thông tin lịch sử tương tác của người dùng (ratings, click logs). Do đó, phương pháp Collaborative Filtering không khả thi. Content-Based Filtering chỉ cần thông tin về bản thân bộ phim.

**Lý do 2 — Bổ sung thông tin đa phương thức:**
Một bộ phim được xác định bởi cả hình ảnh lẫn nội dung văn bản. Hai bộ phim có thể có mô tả văn bản tương tự nhưng hình ảnh khác nhau (hoặc ngược lại). Việc kết hợp cả hai nguồn thông tin cho phép hệ thống nắm bắt được sự tương đồng toàn diện hơn.

**So sánh với các phương pháp khác:**

| Phương pháp | Ưu điểm | Nhược điểm | Ứng dụng trong KhaiPha |
|-------------|---------|-----------|----------------------|
| Collaborative Filtering | Cá nhân hóa cao | Cần dữ liệu người dùng, cold start | Không áp dụng |
| Content-Based (văn bản) | Không cần người dùng | Bỏ qua thông tin hình ảnh | Áp dụng (TF-IDF) |
| Content-Based (hình ảnh) | Phát hiện phim cùng phong cách | Bỏ qua nội dung ngữ nghĩa | Áp dụng (CNN) |
| **Multimodal (KhaiPha)** | **Kết hợp cả hai nguồn** | **Phức tạp hơn, cần nhiều tài nguyên** | **Cốt lõi hệ thống** |
| Hybrid (Matrix Factorization) | Hiệu suất cao | Cần large-scale data, GPU | Hướng phát triển tương lai |

---

## 2.4 Sơ Đồ Tổng Thể Hệ Thống

```
[Dữ liệu thô TMDB]
        |
        v
[Tiền xử lý: làm sạch văn bản, fetch poster]
        |
        +-----------+-----------+
        |                       |
        v                       v
[CNN ResNet50]            [TF-IDF Vectorizer]
(2,048 chiều)              (500 chiều)
        |                       |
        v                       v
[MinMax Normalize]        [MinMax Normalize]
        |                       |
        +-----------+-----------+
                    |
                    v
        [Đặc trưng kết hợp: 2,548 chiều]
                    |
          +---------+---------+
          |         |         |
          v         v         v
     [K-Means]  [Cosine  [Naive Bayes]
     (cluster)  Similarity] (genre pred)
                    |
                    v
           [Top-K Recommendations]
```

*Hình 2.1: Sơ đồ luồng xử lý tổng thể của hệ thống KhaiPha.*
