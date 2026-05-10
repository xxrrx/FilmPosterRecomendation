# Chương 15: Kết Luận

## 15.1 Tóm Tắt Công Trình

Báo cáo này trình bày quá trình thiết kế, triển khai và đánh giá hệ thống gợi ý phim đa phương thức **KhaiPha** — một hệ thống kết hợp đặc trưng hình ảnh từ CNN ResNet50 và đặc trưng văn bản từ TF-IDF để đưa ra gợi ý phim cá nhân hóa dựa trên nội dung (Content-Based).

Hệ thống được xây dựng trên tập dữ liệu **TMDB 5000 Movies** với 4,768 bộ phim sau tiền xử lý, bao gồm 17 thể loại và khoảng thời gian từ 1916 đến 2017. Pipeline kỹ thuật gồm năm module chính:

1. **Thu thập và tiền xử lý dữ liệu:** Hợp nhất hai file CSV, làm sạch văn bản (lowercase, stopword removal, lemmatization), và thu thập ảnh poster qua TMDB API với 20 luồng song song.

2. **Trích xuất đặc trưng hình ảnh:** Sử dụng ResNet50 (pretrained ImageNet, frozen weights) với lớp GAP tạo ra vector 2,048 chiều cho mỗi poster phim.

3. **Trích xuất đặc trưng văn bản:** TF-IDF Vectorizer với sublinear TF scaling, ngram_range=(1,2), max_features=500, tạo ra vector 500 chiều từ mô tả phim và thể loại.

4. **Phân cụm K-Means:** K=20 cụm trên không gian đặc trưng kết hợp 2,548 chiều, hỗ trợ chức năng khám phá phim theo cụm trong giao diện.

5. **Phân loại thể loại bằng Naive Bayes:** Gaussian NB đa nhãn (MultiOutputClassifier) trên đặc trưng CNN, đạt F1-micro=0.4612 và Recall=0.6910.

6. **Khai phá luật kết hợp Apriori:** Phát hiện 12 luật kết hợp có ý nghĩa thống kê (lift > 1.0) trên đồ thị đồng xuất hiện thể loại, với luật mạnh nhất Mystery → Thriller (lift=2.59).

7. **Hệ thống gợi ý:** Cosine similarity trên combined features với alpha blending cho phép điều chỉnh trọng số hình ảnh/văn bản theo ý muốn người dùng.

---

## 15.2 Đóng Góp Chính

### 15.2.1 Đóng Góp Kỹ Thuật

- **Tích hợp đa phương thức (multimodal fusion):** Kết hợp thành công hai nguồn thông tin bổ sung nhau — hình ảnh poster (CNN, 2,048 chiều) và mô tả văn bản (TF-IDF, 500 chiều) — trong một không gian biểu diễn thống nhất 2,548 chiều.

- **Pipeline end-to-end hoàn chỉnh:** Từ dữ liệu thô (CSV + API) đến ứng dụng web có thể sử dụng thực tế, bao gồm backend API (FastAPI) và frontend tương tác (React 18).

- **Tham số điều khiển alpha:** Cơ chế blending với alpha ∈ [0, 1] cho phép người dùng trực tiếp kiểm soát và quan sát ảnh hưởng của từng phương thức — có giá trị giáo dục cao về cách hoạt động của hệ thống.

### 15.2.2 Đóng Góp Học Thuật

- Phân tích chi tiết ưu nhược điểm của từng thuật toán trong bối cảnh dataset TMDB 5000.
- Giải thích rõ ràng tại sao Silhouette Score thấp (-0.003) trong không gian chiều cao là kết quả tất yếu của Curse of Dimensionality, không phải do lỗi triển khai.
- Khai phá 12 luật kết hợp thể loại phim có ý nghĩa điện ảnh rõ ràng, phù hợp với kiến thức nền về cấu trúc thể loại.

---

## 15.3 Nhận Xét Tổng Quan

### 15.3.1 Điểm Thành Công

**Về kỹ thuật:** Hệ thống hoạt động ổn định, response time ~80ms cho mỗi query gợi ý (trên 4,768 phim × 2,548 chiều). Pipeline có thể tái lập (reproducible) với random_state cố định. Code tổ chức rõ ràng theo module.

**Về chất lượng gợi ý:** Kiểm tra định tính cho thấy gợi ý hợp lý trong hầu hết trường hợp — phim Sci-Fi gợi ý Sci-Fi, Animation gợi ý Animation, franchise gợi ý đúng các phim trong franchise.

**Về đa phương thức:** Kết hợp CNN + TF-IDF cho Silhouette Score tốt hơn so với sử dụng riêng lẻ từng phương thức, xác nhận tính bổ sung của hai nguồn thông tin.

### 15.3.2 Hạn Chế Chính

- **Không có cá nhân hóa:** Thiếu dữ liệu người dùng là giới hạn căn bản nhất của hệ thống hiện tại.
- **Dataset cũ:** Không cập nhật phim sau 2017.
- **TF-IDF hạn chế:** Không nắm được ngữ nghĩa sâu của ngôn ngữ tự nhiên.

---

## 15.4 Bài Học Rút Ra

1. **Transfer learning hiệu quả ngay cả không có GPU chuyên dụng:** ResNet50 với trọng số ImageNet cho đặc trưng đủ tốt cho tác vụ gợi ý phim mà không cần fine-tuning.

2. **Chỉ số định lượng cần được diễn giải theo ngữ cảnh:** Silhouette Score -0.003 không có nghĩa là K-Means hoàn toàn thất bại — kết quả định tính vẫn hữu ích.

3. **Tính bổ sung quan trọng hơn tính mạnh:** Hai đặc trưng yếu nhưng bổ sung nhau (CNN + TF-IDF) cho kết quả tốt hơn một đặc trưng mạnh đơn lẻ.

4. **Pragmatism trong lựa chọn công nghệ:** TF-IDF đơn giản hơn BERT nhưng triển khai trong vài phút và cho kết quả đủ tốt cho yêu cầu hiện tại.

---

## 15.5 Lời Kết

Hệ thống gợi ý phim KhaiPha là một triển khai học máy thực tế, kết hợp nhiều kỹ thuật Data Mining và Machine Learning trong một pipeline hoàn chỉnh có thể sử dụng ngay. Tuy còn nhiều hạn chế, hệ thống đã chứng minh tính khả thi của phương pháp đa phương thức trong bài toán gợi ý nội dung, đặt nền móng vững chắc cho các cải tiến trong tương lai hướng tới cá nhân hóa thực sự và học ngữ nghĩa sâu hơn.

Trong bức tranh rộng hơn của Data Mining và Information Retrieval, KhaiPha minh họa một nguyên lý quan trọng: **sự kết hợp thông minh giữa nhiều nguồn thông tin bổ sung nhau luôn cho kết quả tốt hơn việc phụ thuộc vào một nguồn duy nhất**, bất kể nguồn đó mạnh đến đâu.
