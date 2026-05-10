# TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN
## KHOA KHOA HỌC MÁY TÍNH

---

&nbsp;

&nbsp;

# BÁO CÁO ĐỒ ÁN MÔN HỌC
# KHAI PHÁ DỮ LIỆU VÀ ỨNG DỤNG

---

&nbsp;

## HỆ THỐNG GỢI Ý PHIM ĐA PHƯƠNG THỨC
### KẾT HỢP ĐẶC TRƯNG HÌNH ẢNH CNN VÀ VĂN BẢN TF-IDF

---

&nbsp;

&nbsp;

| | |
|---|---|
| **Môn học** | Khai Phá Dữ Liệu (Data Mining) |
| **Giảng viên hướng dẫn** | |
| **Nhóm thực hiện** | |
| **Năm học** | 2024 – 2025 |

---

&nbsp;

&nbsp;

**TP. Hồ Chí Minh, tháng 5 năm 2025**

---

## Tóm Tắt (Abstract)

Báo cáo này trình bày thiết kế và triển khai của hệ thống gợi ý phim **KhaiPha** — một hệ thống kết hợp đặc trưng hình ảnh từ mạng nơ-ron tích chập (Convolutional Neural Network) và đặc trưng văn bản từ TF-IDF để đưa ra gợi ý phim cá nhân hóa.

Hệ thống được xây dựng trên tập dữ liệu **TMDB 5000 Movies** với 4,768 bộ phim sau quá trình tiền xử lý. Kiến trúc của hệ thống bao gồm năm module chính: (1) thu thập và tiền xử lý dữ liệu, (2) trích xuất đặc trưng hình ảnh bằng ResNet50 (2,048 chiều), (3) trích xuất đặc trưng văn bản bằng TF-IDF (500 chiều), (4) phân cụm K-Means (K=20) để nhóm phim có nội dung tương đồng, (5) phân loại thể loại bằng Naive Bayes Gaussian đa nhãn, và (6) khai phá luật kết hợp Apriori trên đồ thị đồng xuất hiện thể loại.

Các gợi ý được tính toán theo độ tương đồng cosine trên không gian đặc trưng kết hợp (2,548 chiều), với tham số alpha cho phép người dùng điều chỉnh trọng số giữa đặc trưng hình ảnh và văn bản. Kết quả thực nghiệm cho thấy mô hình Naive Bayes đạt F1-micro = 0.4612 và Recall = 0.6910; giải thuật Apriori phát hiện 12 luật kết hợp có ý nghĩa thống kê với lift > 1.0.

**Từ khóa:** Hệ thống gợi ý phim, CNN, ResNet50, TF-IDF, K-Means, Naive Bayes, Apriori, Cosine Similarity, TMDB.
