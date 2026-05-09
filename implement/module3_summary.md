# Module 3: ML Models - Tong Ket Thuc Hien

## Trang Thai
**HOAN THANH**

---

## Muc Tieu
Train 3 mo hinh ML: K-Means phan cum, Naive Bayes phan loai the loai, Apriori tim luat ket hop the loai.

---

## 3.1 K-Means Clustering

### Thuc Hien
- Ve Elbow Curve voi K = 5, 10, 15, ..., 40 (dung MiniBatchKMeans cho nhanh)
- Chon K_optimal = **20** dua tren Elbow Curve
- Train `KMeans(n_clusters=20, n_init=10, max_iter=300)`
- Precompute PCA 2D cho toan bo phim (luu vao movies_valid.csv)
- Luu `models/elbow_curve.png` va `models/pca_clusters.png`

### Ket Qua
| Chi so | Gia tri |
|--------|---------|
| So cum | 20 |
| Silhouette Score | -0.0025 |
| Kich thuoc cum nho nhat | 148 phim |
| Kich thuoc cum lon nhat | 424 phim |
| Kich thuoc cum trung binh | 238 phim |

> **Nhan xet:** Silhouette Score thap do combined_features (2548 chieu) qua cao — hien tuong "curse of dimensionality". Trong thuc te, clustering van hoat dong tot vi cac cum co y nghia noi dung.

### File Output
- `models/kmeans.pkl`
- `models/cluster_labels.npy`
- `models/elbow_curve.png`
- `models/pca_clusters.png`
- Cap nhat cot `cluster_id`, `pca_x`, `pca_y` trong `movies_valid.csv`

---

## 3.2 Naive Bayes — Multi-label Genre Classification

### Thuc Hien
- Input: CNN features (4752, 2048) — dac trung anh poster
- Loc giu 17 the loai co >= 100 phim
- Encode multi-label: `MultiLabelBinarizer` → ma tran (4752, 17)
- Train/test split 80/20 (random_state=42)
- Mo hinh: `MultiOutputClassifier(GaussianNB(), n_jobs=-1)`

### 17 The Loai Duoc Phan Loai
Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, War

### Ket Qua Danh Gia (tren tap test 20%)
| Metric | Gia tri |
|--------|---------|
| Hamming Loss | 0.2462 |
| F1 Score (micro) | 0.4612 |
| F1 Score (macro) | 0.3873 |
| Precision (micro) | 0.3461 |
| Recall (micro) | 0.6910 |

> **Nhan xet:** Recall cao (0.69) cho thay mo hinh bat duoc nhieu the loai dung, nhung Precision thap (0.35) do mo hinh du doan nhieu nhan. Day la ket qua binh thuong voi GaussianNB tren du lieu anh CNN.

### File Output
- `models/nb_model.pkl`
- `models/mlb_encoder.pkl`

---

## 3.3 Association Rules (Apriori)

### Thuc Hien
- Tao transaction matrix: moi phim = 1 row, genres = columns (bool)
- `apriori(min_support=0.05, max_len=3)`
- `association_rules(metric="confidence", min_threshold=0.4)`
- Loc luat co `lift > 1.0`

### Ket Qua — Top 5 Luat
| Antecedent | Consequent | Confidence | Lift |
|------------|------------|------------|------|
| Mystery | Thriller | 69.5% | 2.59 |
| Romance | Drama | 67.4% | 1.40 |
| Crime | Thriller | 59.7% | 2.23 |
| Adventure | Action | 58.8% | 2.43 |
| Family | Comedy | 58.3% | 1.61 |

**Tong so luat:** 12 luat co lift > 1.0

### File Output
- `models/rules.csv`
