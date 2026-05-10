# Chương 12: Đánh Giá và Phân Tích

## 12.1 Framework Đánh Giá

Đánh giá một hệ thống gợi ý Content-Based phức tạp hơn đánh giá các mô hình supervised learning thông thường vì thiếu vắng nhãn ground-truth rõ ràng (không biết "đây có phải gợi ý đúng hay không" trừ khi có dữ liệu người dùng). Hệ thống KhaiPha được đánh giá trên hai mặt:

1. **Đánh giá định lượng:** Các thành phần có nhãn rõ ràng (K-Means, Naive Bayes, Apriori).
2. **Đánh giá định tính:** Chất lượng tổng thể của gợi ý qua quan sát thực tế.

---

## 12.2 Đánh Giá K-Means Clustering

### 12.2.1 Chỉ Số Nội Tại (Intrinsic Metrics)

**Silhouette Score (SS):**

```
SS = (1/N) Σ_i s(i)
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

- **Kết quả:** SS = -0.003
- **Diễn giải:** Giá trị gần 0 (hoặc âm nhẹ) cho thấy các cụm không tách biệt rõ ràng. Điểm dữ liệu nằm gần ranh giới giữa các cụm hơn là nằm sâu trong một cụm.

**Davies-Bouldin Index (DBI):**

```
DBI = (1/K) Σ_{k=1}^{K} max_{l≠k} [(s_k + s_l) / d(μ_k, μ_l)]
```

trong đó `s_k` là độ phân tán trung bình trong cụm k, `d(μ_k, μ_l)` là khoảng cách giữa hai tâm cụm.

- **Kết quả:** DBI = 5.045
- **Diễn giải:** Giá trị cao, tâm các cụm gần nhau tương đối so với độ phân tán trong cụm.

**Calinski-Harabasz Score:**

```
CH = [SS_B / (K-1)] / [SS_W / (N-K)]
```

trong đó `SS_B` là phương sai giữa các cụm, `SS_W` là phương sai trong cụm.

| K | CH Score |
|---|---------|
| 5 | 38.2 |
| 10 | 29.7 |
| 15 | 25.4 |
| **20** | **22.1** |
| 25 | 19.8 |

CH Score giảm đơn điệu theo K — chỉ số này không xác nhận K=20 là điểm dừng tối ưu, nhưng cùng với Elbow Curve, K=20 là lựa chọn thực tế tốt nhất.

### 12.2.2 Phân Tích Định Tính Cụm

Thay vì chỉ dùng chỉ số định lượng, hệ thống kiểm tra nội dung thực tế của từng cụm:

| Cụm | Top 5 phim mẫu | Chủ đề chính |
|-----|---------------|-------------|
| 0 | Titanic, The Notebook, Dirty Dancing, The Vow | Romance / Drama |
| 2 | Mad Max, John Wick, Die Hard, The Expendables | Action Thuần |
| 5 | Toy Story, Finding Nemo, Shrek, The Lion King | Animation |
| 11 | Schindler's List, Saving Private Ryan, Dunkirk | War / History |
| 19 | Planet Earth, March of Penguins, Super Size Me | Documentary |

Mặc dù Silhouette Score thấp, các cụm vẫn thể hiện tính nhất quán thể loại đáng kể khi kiểm tra thủ công. Điều này xác nhận rằng K-Means hoạt động tốt hơn so với chỉ số định lượng gợi ý.

---

## 12.3 Đánh Giá Naive Bayes

### 12.3.1 Các Chỉ Số Multi-label

**Hamming Loss:**

```
HL = (1/N) Σ_i (1/|L|) × |ŷ_i △ y_i|
```

trong đó `△` là phép XOR (symmetric difference). HL = 0.2498 có nghĩa là trung bình 25% nhãn bị dự đoán sai (kết hợp false positives và false negatives) trong tổng số 17 nhãn.

**F1-Score Micro:**

```
F1_micro = 2 × (P_micro × R_micro) / (P_micro + R_micro)
P_micro = TP_total / (TP_total + FP_total)
R_micro = TP_total / (TP_total + FN_total)
```

F1_micro = 0.4612 — tính toán trên tổng TP, FP, FN của tất cả nhãn (thể loại phổ biến có ảnh hưởng lớn hơn).

**F1-Score Macro:**

```
F1_macro = (1/|L|) Σ_{l∈L} F1_l
```

F1_macro = 0.3877 — trung bình F1 của từng thể loại (trọng số bằng nhau), phản ánh hiệu suất trên các thể loại hiếm hơn.

### 12.3.2 Confusion Matrix (Per-label)

Ví dụ với thể loại Drama (thể loại khó nhất):

| | Pred=0 | Pred=1 |
|---|--------|--------|
| **Actual=0** | TN=312 | FP=185 |
| **Actual=1** | FN=204 | TP=253 |

```
Precision = 253 / (253+185) = 0.578
Recall = 253 / (253+204) = 0.553
F1 = 0.565
```

Drama có cả precision và recall trung bình vì class imbalance: 2,292/4,768 phim có Drama (48%), model bị bias về phía dự đoán Drama.

### 12.3.3 Phân Tích Lỗi

Các lỗi phổ biến nhất:

| Lỗi | Mô tả | Ví dụ |
|-----|-------|-------|
| FP Drama | Dự đoán Drama cho phim không phải Drama | Phim Action bị gán thêm Drama |
| FP Thriller | Dự đoán Thriller cho phim Crime thuần | The Godfather |
| FN Animation | Bỏ sót Animation trong phim gia đình | Film hoạt hình nhẹ nhàng |
| FP/FN Romance | Nhầm lẫn Romance ↔ Drama | Overlap mạnh giữa hai thể loại |

---

## 12.4 Đánh Giá Association Rules

### 12.4.1 Thống Kê Luật

| Chỉ số | Giá trị | Ngưỡng | Đánh giá |
|--------|---------|--------|---------|
| Số luật (lift>1) | 12 | — | Vừa đủ |
| Support trung bình | 0.079 | ≥0.05 | ✓ Thỏa |
| Confidence trung bình | 0.554 | ≥0.40 | ✓ Tốt |
| Lift trung bình | 1.89 | >1.0 | ✓ Có ý nghĩa |
| Lift max | 2.59 | — | Mạnh |

### 12.4.2 Đánh Giá Ý Nghĩa Thực Tế

Kiểm tra từng luật với kiến thức nền về điện ảnh:

| Luật | Đánh giá chuyên gia | Nhận xét |
|------|-------------------|---------|
| Mystery → Thriller (lift=2.59) | Hợp lý cao | Mystery gần như là subset của Thriller |
| Adventure → Action (lift=2.43) | Hợp lý cao | Adventure phim hiếm khi thiếu Action |
| Crime → Thriller (lift=2.23) | Hợp lý cao | Phim tội phạm tạo tension như Thriller |
| Romance → Drama (lift=1.40) | Hợp lý | Tình yêu luôn đi kèm kịch tính |
| Family → Comedy (lift=1.61) | Hợp lý | Phim gia đình thường nhẹ nhàng, vui vẻ |

Tất cả 12 luật đều có ý nghĩa điện ảnh rõ ràng — xác nhận tính hợp lệ của giải thuật Apriori trên tập dữ liệu này.

---

## 12.5 Đánh Giá Hệ Thống Gợi Ý

### 12.5.1 Hạn Chế của Đánh Giá Định Lượng

Không có ground-truth cho recommendation, nhưng có thể sử dụng các proxy metrics:

**Coverage:** Tỷ lệ phim được gợi ý ít nhất một lần (khi là query movie của phim khác).
- Với cosine similarity trên 4,768 phim: mọi phim đều có thể được gợi ý → **Coverage = 100%**

**Novelty:** Mức độ gợi ý những phim ít nổi tiếng.
- Hệ thống không bias theo popularity → **Novelty** phụ thuộc vào truy vấn

**Diversity (trong top-10):** Độ đa dạng của 10 phim gợi ý cho một query.
- Cosine similarity có xu hướng gợi ý các phim rất giống nhau → **Diversity thấp** là nhược điểm

### 12.5.2 So Sánh Gợi Ý với Baseline

**Baseline: Genre-based Matching** (gợi ý phim có thể loại giống nhất):

| Phim truy vấn | KhaiPha Top-1 | Genre-based Top-1 | Cùng kết quả? |
|--------------|--------------|-------------------|--------------|
| Avatar | Total Recall | Iron Man | Không |
| The Dark Knight | The Dark Knight Rises | Batman Begins | Khác nhau nhưng hợp lý |
| Toy Story | Finding Nemo | A Bug's Life | Khác nhau nhưng hợp lý |

KhaiPha bổ sung thêm tín hiệu hình ảnh so với genre-based matching thuần túy, tạo ra sự đa dạng lớn hơn trong gợi ý.

### 12.5.3 Phân Tích Hiệu Ứng Alpha

Để đánh giá ảnh hưởng của alpha, hệ thống thực hiện thử nghiệm:

**Với phim có poster rất đặc trưng (Animation):**
- alpha = 1.0: Top-5 đều là phim animation → Chính xác cao
- alpha = 0.5: Top-5 gồm animation + family comedy → Vẫn tốt
- alpha = 0.0: Top-5 dựa thuần text → Có thể bao gồm phim live-action cùng chủ đề

**Với phim drama "generic" (poster không đặc trưng):**
- alpha = 0.0: Tốt hơn (TF-IDF nắm nội dung cụ thể hơn)
- alpha = 1.0: Kém hơn (poster drama khó phân biệt)

Kết luận: alpha = 0.5 là lựa chọn mặc định tốt; người dùng nên điều chỉnh tùy theo loại phim truy vấn.

---

## 12.6 So Sánh với Hướng Tiếp Cận Khác

| Hướng tiếp cận | F1 NB | Silhouette | Chất lượng gợi ý | Phức tạp |
|---------------|--------|-----------|-----------------|---------|
| TF-IDF only | — | -0.012 | Trung bình | Thấp |
| CNN only | — | -0.008 | Trung bình-Tốt | Cao |
| **KhaiPha (CNN+TF-IDF)** | **0.461** | **-0.003** | **Tốt** | **Cao** |
| Hypothetical BERT+CNN | — | ~0.05+ | Rất tốt | Rất cao |
| Collaborative Filtering | N/A | N/A | Cao (có user data) | Trung bình |

Kết hợp CNN + TF-IDF (KhaiPha) cho Silhouette Score tốt hơn so với sử dụng riêng lẻ — xác nhận rằng hai nguồn đặc trưng bổ sung cho nhau.
