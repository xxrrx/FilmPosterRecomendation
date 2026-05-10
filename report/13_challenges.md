# Chương 13: Thách Thức và Giới Hạn

## 13.1 Thách Thức Kỹ Thuật

### 13.1.1 Curse of Dimensionality

Không gian đặc trưng kết hợp 2,548 chiều đặt ra thách thức nghiêm trọng cho các thuật toán dựa trên khoảng cách như K-Means và Cosine Similarity.

**Vấn đề:** Trong không gian chiều cao, tất cả cặp điểm có xu hướng trở nên "xa nhau xấp xỉ như nhau" (concentration of measure phenomenon). Cụ thể, tỷ lệ giữa khoảng cách lớn nhất và nhỏ nhất tiến về 1:

```
lim_{d→∞} [max_dist(X) - min_dist(X)] / min_dist(X) → 0
```

**Hậu quả thực tế:**
- Silhouette Score = -0.003 (gần như tệ như phân cụm ngẫu nhiên)
- Cosine similarity giữa các cặp phim ngẫu nhiên thường cao hơn 0.3 (ít phân biệt)
- K-Means khó tìm được cụm có ranh giới rõ ràng

**Giải pháp đã áp dụng:** Giảm chiều bằng PCA (2D cho visualization) nhưng chỉ giữ ~12% phương sai.

**Giải pháp tiềm năng:** UMAP hoặc t-SNE có thể giữ cấu trúc cục bộ tốt hơn PCA cho visualization; Autoencoder có thể học biểu diễn nén phi tuyến tính hiệu quả hơn.

---

### 13.1.2 Giả Định Gaussian Không Phù Hợp trong Naive Bayes

Gaussian Naive Bayes giả định đặc trưng CNN tuân theo phân phối chuẩn có điều kiện:

```
P(x_j | y=k) ~ N(μ_{k,j}, σ^2_{k,j})
```

**Thực tế:** Đặc trưng CNN từ lớp ReLU có phân phối lệch phải nghiêm trọng (right-skewed) với nhiều giá trị bằng 0 (sparse activation). Phân phối này vi phạm giả định Gaussian.

**Hậu quả:** Ước lượng xác suất `P(x_j | y=k)` không chính xác, ảnh hưởng đến độ tin cậy của dự đoán. Tuy nhiên, Naive Bayes vẫn đủ chính xác để phân loại (vì chỉ cần thứ tự tương đối của xác suất, không cần giá trị tuyệt đối).

**Giải pháp:** Chuyển sang Multinomial NB với histogram features, hoặc dùng Linear SVM / Random Forest.

---

### 13.1.3 Class Imbalance trong Phân Loại Thể Loại

Drama chiếm 48% dataset (2,292/4,768 phim), trong khi Documentary chỉ chiếm 2.9% (138 phim) — tỷ lệ mất cân bằng gần 17:1.

**Hậu quả:**
- Model thiên về dự đoán Drama (majority class)
- Exact Match Accuracy thấp: chỉ 1.05%
- F1-macro (0.3877) thấp hơn F1-micro (0.4612) — cho thấy model hoạt động tệ hơn trên thể loại hiếm

**Giải pháp tiềm năng:**
1. **Class weights:** Tăng penalty cho minority classes trong loss function
2. **SMOTE / Oversampling:** Tăng mẫu cho thể loại hiếm
3. **Threshold tuning:** Hạ ngưỡng classification (< 0.5) cho thể loại hiếm

---

### 13.1.4 Chất Lượng Poster Không Đồng Đều

Poster phim có chất lượng rất khác nhau:
- Phim mới (2000+): Poster chất lượng cao, màu sắc sống động, thiết kế chuyên nghiệp
- Phim cũ (trước 1980): Poster được scan từ bản in, độ phân giải thấp, màu sắc phai
- Phim ngoài tiếng Anh: Đôi khi dùng poster địa phương với bố cục khác hẳn

**Hậu quả:** CNN học được đặc trưng không nhất quán giữa các thời kỳ. Phim cổ điển ít có khả năng được gợi ý cùng phim hiện đại dù có nội dung tương đồng.

---

### 13.1.5 Văn Bản Overview Ngắn

Với trung bình 29.8 từ/phim sau làm sạch (thực tế ~20 từ sau stopword removal), TF-IDF có rất ít "bề mặt ngữ nghĩa" để hoạt động.

**Hậu quả:**
- Nhiều phim có vector TF-IDF thưa (sparse) cực kỳ, dẫn đến cosine similarity thấp với hầu hết phim khác
- Bigrams ít có cơ hội xuất hiện đủ thường để vào top-500 features
- TF-IDF không thể nắm bắt được ngữ nghĩa câu phức tạp

---

## 13.2 Giới Hạn Dữ Liệu

### 13.2.1 Không Có Dữ Liệu Người Dùng

Hệ thống hoàn toàn dựa trên nội dung phim (Content-Based). Không có:
- Lịch sử xem của người dùng
- Ratings cá nhân
- Click logs hay watch time

**Hậu quả:** Không thể cá nhân hóa gợi ý theo từng người dùng. Hai người dùng với sở thích hoàn toàn khác nhau nhận được cùng gợi ý cho cùng một phim truy vấn.

### 13.2.2 Dataset Tĩnh (Cold Start Problem cho Phim Mới)

Tập dữ liệu chỉ bao gồm phim đến năm 2017. Phim ra mắt sau năm 2017 không thể được gợi ý trừ khi:
1. Thêm phim vào database
2. Trích xuất đặc trưng CNN + TF-IDF
3. Re-compute similarity matrix

**Hậu quả:** Hệ thống không phản ánh xu hướng điện ảnh hiện đại (2018–2025).

### 13.2.3 Thiên Lệch theo Thị Trường Anh Ngữ

97% phim trong dataset là phim tiếng Anh hoặc có overview tiếng Anh. Phim Á, Âu lục địa, và khu vực khác rất ít đại diện.

**Hậu quả:**
- TF-IDF được huấn luyện chủ yếu trên từ vựng điện ảnh Anh Mỹ
- Gợi ý thiên về phim Hollywood

---

## 13.3 Giới Hạn Mô Hình

### 13.3.1 K-Means: Giả Định Cụm Hình Cầu

K-Means tối ưu hóa khoảng cách Euclidean bình phương — giả định các cụm có dạng hình cầu (spherical clusters) với kích thước tương đương. Trong thực tế, các nhóm phim có hình dạng phức tạp và kích thước không đồng đều.

### 13.3.2 Cosine Similarity: Bỏ Qua Độ Lớn Vector

Cosine similarity chỉ đo góc giữa hai vector, bỏ qua độ lớn. Hai phim có vector đặc trưng rất nhỏ (tức là đặc trưng yếu ớt, không đặc trưng) nhưng cùng hướng sẽ có similarity = 1.0 — điều này có thể tạo ra gợi ý "giả" tốt.

### 13.3.3 Alpha Blending: Không Học Được Trọng Số Tối Ưu

Trọng số alpha được người dùng chọn thủ công, không được học tự động từ feedback. Không có cơ chế nào để xác định alpha tối ưu cho từng cặp (query movie, candidate movie).

---

## 13.4 Giới Hạn Hạ Tầng

### 13.4.1 Không Hỗ Trợ Scale

- SQLite không phù hợp cho >100,000 phim
- NumPy cosine similarity là O(N × d) cho mỗi query — với N = triệu phim, đây là bottleneck nghiêm trọng
- Không có caching layer (Redis), mỗi request tính lại similarity hoàn toàn

### 13.4.2 Thiếu Authentication và User Management

Backend hiện tại không có:
- Hệ thống đăng nhập người dùng
- Lưu lịch sử tìm kiếm/xem
- Rate limiting cho API

---

## 13.5 Bảng Tóm Tắt Thách Thức và Hướng Giải Quyết

| Thách thức | Mức độ ảnh hưởng | Hướng giải quyết |
|-----------|-----------------|-----------------|
| Curse of Dimensionality | Cao | UMAP, Autoencoder |
| Gaussian assumption vi phạm | Trung bình | Random Forest, SVM |
| Class imbalance | Trung bình | Oversampling, class weights |
| Overview ngắn | Trung bình | Sentence-BERT |
| Không có user data | Cao | Thu thập feedback, Hybrid CF |
| Dataset tĩnh (2017) | Cao | Pipeline cập nhật tự động |
| Thiên lệch Anh Mỹ | Thấp-Trung bình | Mở rộng dataset đa ngôn ngữ |
| Không scale được | Trung bình | FAISS, PostgreSQL |
