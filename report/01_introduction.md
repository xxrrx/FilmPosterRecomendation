# Chương 1: Giới Thiệu

## 1.1 Bối Cảnh và Động Lực

Trong kỷ nguyên số hóa hiện nay, khối lượng nội dung giải trí trực tuyến tăng trưởng theo cấp số nhân. Chỉ riêng nền tảng Netflix đã sở hữu hơn 15,000 đầu phim và chương trình, trong khi các nền tảng như Amazon Prime Video, Disney+, và HBO Max cộng lại con số này lên đến hàng trăm nghìn tựa phim. Trước sự bùng nổ thông tin đó, việc một người dùng tự tìm kiếm bộ phim phù hợp với sở thích cá nhân trở nên cực kỳ tốn thời gian và thiếu hiệu quả — hiện tượng này thường được gọi là **information overload** (quá tải thông tin).

Hệ thống gợi ý (Recommender System) ra đời như một giải pháp tất yếu, cho phép tự động hóa quá trình lọc và đề xuất nội dung phù hợp với từng người dùng. Theo nghiên cứu của McKinsey (2013), hơn 75% lượng nội dung được xem trên Netflix đến từ hệ thống gợi ý của nền tảng này. Con số đó phản ánh tầm quan trọng chiến lược của Recommender Systems trong kinh tế nội dung số.

Tuy nhiên, phần lớn các hệ thống gợi ý truyền thống tập trung vào một loại thông tin duy nhất — hoặc là lịch sử xem (Collaborative Filtering), hoặc là metadata văn bản của phim (Content-Based Filtering dựa trên mô tả, thể loại). Cách tiếp cận đơn phương thức này bỏ qua một nguồn tín hiệu phong phú: **hình ảnh trực quan của poster phim**. Poster là biểu trưng thẩm mỹ của một bộ phim — màu sắc, bố cục, hình ảnh nhân vật, phong cách đồ họa — tất cả chứa đựng thông tin ngữ nghĩa sâu sắc về thể loại, không khí, và đối tượng khán giả mục tiêu.

Đồ án **KhaiPha** được xây dựng nhằm giải quyết khoảng trống này bằng cách thiết kế một hệ thống gợi ý **đa phương thức** (multimodal), kết hợp đặc trưng hình ảnh từ mạng CNN với đặc trưng văn bản từ TF-IDF trong một không gian biểu diễn thống nhất.

---

## 1.2 Mục Tiêu Đề Tài

Đề tài đặt ra các mục tiêu cụ thể sau:

**Mục tiêu kỹ thuật:**
1. Xây dựng pipeline thu thập và tiền xử lý dữ liệu từ tập dữ liệu TMDB 5000 Movies.
2. Trích xuất đặc trưng hình ảnh từ poster phim bằng mạng ResNet50 (transfer learning).
3. Trích xuất đặc trưng văn bản từ mô tả và thể loại phim bằng TF-IDF.
4. Áp dụng thuật toán K-Means để phân cụm phim theo nội dung tổng hợp.
5. Huấn luyện mô hình Naive Bayes đa nhãn để dự đoán thể loại từ đặc trưng hình ảnh.
6. Khai phá luật kết hợp Apriori trên đồ thị đồng xuất hiện thể loại phim.
7. Xây dựng API gợi ý sử dụng độ tương đồng cosine với tham số điều chỉnh alpha.

**Mục tiêu sản phẩm:**
1. Triển khai backend RESTful API bằng FastAPI.
2. Triển khai giao diện web người dùng bằng React 18.
3. Cho phép tìm kiếm, duyệt phim, và xem gợi ý theo thời gian thực.

**Mục tiêu học thuật:**
1. Đánh giá hiệu quả của từng thuật toán bằng các chỉ số phù hợp.
2. Phân tích ưu nhược điểm và giới hạn của hệ thống.
3. Đề xuất hướng cải thiện trong tương lai.

---

## 1.3 Phạm Vi và Giới Hạn

Đề tài giới hạn trong phạm vi sau:

- **Dữ liệu:** Tập dữ liệu tĩnh TMDB 5000 (không cập nhật thời gian thực).
- **Số lượng phim:** 4,768 phim sau lọc (từ 4,803 phim gốc).
- **Loại gợi ý:** Gợi ý dựa trên nội dung (Content-Based) — không có thông tin người dùng.
- **Ngôn ngữ:** Tất cả mô tả phim bằng tiếng Anh (từ TMDB API).
- **Phần cứng:** Pipeline được thiết kế chạy trên CPU; GPU tùy chọn cho bước CNN.

---

## 1.4 Cấu Trúc Báo Cáo

Báo cáo được tổ chức theo cấu trúc sau:

| Chương | Nội dung |
|--------|----------|
| Chương 2 | Phát biểu bài toán và định nghĩa hình thức |
| Chương 3 | Mô tả tập dữ liệu TMDB 5000 |
| Chương 4 | Tiền xử lý dữ liệu |
| Chương 5 | Trích xuất đặc trưng hình ảnh bằng CNN ResNet50 |
| Chương 6 | Trích xuất đặc trưng văn bản bằng TF-IDF |
| Chương 7 | Phân cụm K-Means |
| Chương 8 | Phân loại thể loại bằng Naive Bayes |
| Chương 9 | Khai phá luật kết hợp Apriori |
| Chương 10 | Hệ thống gợi ý tổng thể |
| Chương 11 | Kết quả thực nghiệm |
| Chương 12 | Đánh giá và phân tích |
| Chương 13 | Thách thức và giới hạn |
| Chương 14 | Hướng phát triển tương lai |
| Chương 15 | Kết luận |

---

## 1.5 Công Nghệ Sử Dụng

Hệ thống được xây dựng trên nền tảng các thư viện và framework sau:

**Ngôn ngữ lập trình:** Python 3.9+, JavaScript (ES2022)

**Học máy và xử lý dữ liệu:**
- TensorFlow 2.21 / Keras — xây dựng và inference mô hình CNN
- scikit-learn 1.3 — K-Means, Naive Bayes, TF-IDF, đánh giá mô hình
- mlxtend — thuật toán Apriori và sinh luật kết hợp
- NLTK — tiền xử lý văn bản (lemmatization, stopword removal)
- NumPy, Pandas — xử lý ma trận và bảng dữ liệu

**Backend:**
- FastAPI — web framework hiệu năng cao cho REST API
- SQLAlchemy + SQLite — ORM và cơ sở dữ liệu nhẹ
- Uvicorn — ASGI server

**Frontend:**
- React 18 — thư viện giao diện người dùng
- Vite — build tool
- Tailwind CSS — utility-first CSS framework
- Recharts — thư viện biểu đồ cho React

**Nguồn dữ liệu:**
- Kaggle TMDB 5000 Movies Dataset
- TMDB API v3 — lấy ảnh poster

![Hình 1.1: Kiến trúc tổng thể hệ thống KhaiPha](../output/architecture_overview.png)

*Hình 1.1: Sơ đồ kiến trúc tổng thể hệ thống KhaiPha, từ dữ liệu thô đến giao diện người dùng cuối.*
