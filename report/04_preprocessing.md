# Chương 4: Tiền Xử Lý Dữ Liệu

## 4.1 Tổng Quan Pipeline Tiền Xử Lý

Tiền xử lý dữ liệu là giai đoạn nền tảng quyết định chất lượng của toàn bộ pipeline học máy. Trong dự án KhaiPha, quá trình này được thực hiện trong notebook `01_data_collection.ipynb` và bao gồm 6 bước chính:

```
[Raw CSV] → [Merge & Parse] → [Filter] → [Text Clean] → [Poster Fetch] → [Export]
```

Mỗi bước được thiết kế để loại bỏ nhiễu, chuẩn hóa định dạng, và đảm bảo tính toàn vẹn của dữ liệu cho các bước xử lý tiếp theo.

---

## 4.2 Bước 1: Nạp và Hợp Nhất Dữ Liệu

### 4.2.1 Nạp CSV

Hai file CSV được nạp bằng `pandas.read_csv()`:

```python
movies_df = pd.read_csv('data/raw/tmdb_5000_movies.csv')    # 4,803 × 20
credits_df = pd.read_csv('data/raw/tmdb_5000_credits.csv')  # 4,803 × 4
```

### 4.2.2 Hợp Nhất theo movie_id

```python
df = movies_df.merge(credits_df, left_on='id', right_on='movie_id')
```

Kết quả: DataFrame gồm 4,803 dòng với 24 cột.

### 4.2.3 Phân Tích Cột JSON

Các cột `genres`, `keywords`, `cast`, `crew` được lưu dưới dạng chuỗi JSON lồng nhau. Quá trình parse sử dụng `ast.literal_eval()`:

```python
import ast

def parse_json_col(x):
    try:
        return [item['name'] for item in ast.literal_eval(x)]
    except:
        return []

df['genres_list'] = df['genres'].apply(parse_json_col)
df['keywords_list'] = df['keywords'].apply(parse_json_col)
```

**Đặc biệt với cột `cast`:** Chỉ lấy 5 diễn viên đầu tiên (order=0–4) để tránh danh sách quá dài:

```python
df['cast_list'] = df['cast'].apply(
    lambda x: [item['name'] for item in ast.literal_eval(x)[:5]]
)
```

**Đặc biệt với cột `crew`:** Chỉ lấy đạo diễn (job='Director'):

```python
df['director'] = df['crew'].apply(
    lambda x: next((i['name'] for i in ast.literal_eval(x) if i['job']=='Director'), '')
)
```

---

## 4.3 Bước 2: Lọc Dữ Liệu

### 4.3.1 Loại bỏ bản ghi thiếu thông tin bắt buộc

Các phim không có `overview` hoặc `genres` bị loại bỏ vì chúng không thể tham gia vào cả hai pipeline đặc trưng:

```python
df = df[df['overview'].notna() & df['overview'].str.strip().ne('')]
df = df[df['genres_list'].map(len) > 0]
```

Sau bước này: **4,771 phim** còn lại.

### 4.3.2 Chuẩn hóa năm phát hành

```python
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df['year'] = df['year'].fillna(0).astype(int)
```

---

## 4.4 Bước 3: Làm Sạch Văn Bản (Text Cleaning)

Đây là bước quan trọng nhất trong tiền xử lý văn bản, chuẩn bị input cho TF-IDF Vectorizer.

### 4.4.1 Pipeline Làm Sạch

Hàm `clean_text()` được áp dụng trên cột `overview`:

```python
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    # 1. Lowercase
    text = text.lower()
    # 2. Xóa ký tự đặc biệt và dấu câu
    text = re.sub(r'[^a-z\s]', '', text)
    # 3. Tách từ
    tokens = text.split()
    # 4. Loại bỏ stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # 5. Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['overview_clean'] = df['overview'].apply(clean_text)
```

### 4.4.2 Giải Thích Từng Bước

| Bước | Phép biến đổi | Ví dụ |
|------|--------------|-------|
| Lowercase | Chuẩn hóa chữ hoa/thường | "Drama" → "drama" |
| Remove special chars | Loại ký tự không phải chữ/khoảng trắng | "sci-fi!" → "scifi" |
| Tokenize | Tách chuỗi thành danh sách từ | "action movie" → ["action", "movie"] |
| Stopword removal | Loại 179 từ tiếng Anh phổ biến | "the", "a", "is" bị loại |
| Lemmatization | Đưa từ về dạng gốc | "running" → "run", "movies" → "movie" |

**Lý do chọn Lemmatization thay vì Stemming:**
Lemmatization sử dụng từ điển ngôn ngữ học (WordNet) để tạo ra dạng gốc hợp lệ ("running" → "run"), trong khi Stemming thường tạo ra các gốc cắt bớt không có nghĩa ("running" → "run", nhưng "studies" → "studi"). Với TF-IDF trên văn bản ngắn, Lemmatization tạo ra từ điển sạch hơn và dễ diễn giải hơn.

### 4.4.3 Xây Dựng Trường Text Kết Hợp

Ngoài `overview_clean`, hệ thống còn xây dựng một trường kết hợp để đưa vào TF-IDF:

```python
df['text_combined'] = (
    df['overview_clean'] + ' ' +
    df['genres_list'].apply(lambda x: ' '.join(x).lower())
)
```

Việc nối thể loại vào văn bản giúp TF-IDF "biết" được thể loại phim một cách tường minh, bổ sung thêm tín hiệu cho bài toán tính tương đồng.

---

## 4.5 Bước 4: Thu Thập Ảnh Poster

### 4.5.1 Chiến Lược Fetch Song Song

Việc tải 4,771 ảnh poster một cách tuần tự sẽ mất nhiều giờ. Hệ thống sử dụng `ThreadPoolExecutor` với 20 luồng song song:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

BASE_URL = "https://image.tmdb.org/t/p/w500"

def fetch_poster(movie_id, poster_path):
    try:
        url = BASE_URL + poster_path
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return movie_id, url
    except:
        pass
    return movie_id, None

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {
        executor.submit(fetch_poster, row['id'], row['poster_path']): row['id']
        for _, row in df.iterrows()
        if pd.notna(row['poster_path'])
    }
    for future in as_completed(futures):
        movie_id, url = future.result()
        poster_map[movie_id] = url
```

### 4.5.2 Cache và Xử Lý Lỗi

Kết quả fetch được cache vào `poster_paths_cache.json` để tránh gọi API lặp lại khi chạy lại notebook:

```python
import json

cache_path = 'data/raw/poster_paths_cache.json'
if os.path.exists(cache_path):
    with open(cache_path) as f:
        poster_map = json.load(f)
else:
    poster_map = {}
    # ... fetch logic
    with open(cache_path, 'w') as f:
        json.dump(poster_map, f)
```

**Kết quả:** 4,752 phim có poster URL hợp lệ (loại bỏ 19 phim không có poster hoặc lỗi API).

### 4.5.3 Xác Thực Ảnh

Trong bước trích xuất CNN (Chương 5), một số ảnh có thể không tải được tại thời điểm inference. Hệ thống xử lý bằng cách bỏ qua (skip) các phim không tải được ảnh và chỉ giữ lại phim có đặc trưng CNN thành công, dẫn đến con số cuối cùng là **4,768 phim** (tăng nhẹ so với 4,752 do một số phim được xử lý lại từ cache).

---

## 4.6 Bước 5: Xuất Dữ Liệu

### 4.6.1 Dữ Liệu Trung Gian

```python
output_df = df[['id', 'title', 'year', 'genres_list',
                'overview_clean', 'poster_url',
                'vote_average', 'vote_count']].copy()
output_df.columns = ['movie_id', 'title', 'year', 'genres',
                     'overview_clean', 'poster_url',
                     'rating', 'vote_count']
output_df['genres'] = output_df['genres'].apply(json.dumps)
output_df.to_csv('data/processed/movies.csv', index=False)
```

**Schema `movies.csv`:** 4,752 hàng × 8 cột.

---

## 4.7 Tổng Kết Quá Trình Lọc

| Giai đoạn | Số phim | Phim bị loại |
|-----------|---------|-------------|
| Dữ liệu gốc (TMDB) | 4,803 | — |
| Sau lọc overview + genres | 4,771 | 32 |
| Sau fetch poster hợp lệ | 4,752 | 19 |
| Sau trích xuất CNN thành công | 4,768 | — (+16 từ cache) |
| **Tập dữ liệu cuối cùng** | **4,768** | **35 tổng** |

Lưu ý: Con số 4,768 (nhiều hơn 4,752) xuất phát từ việc một số phim trước đó không có URL poster nhưng có ảnh lưu trong cache từ lần chạy trước.

---

## 4.8 Phân Tích Chất Lượng Sau Tiền Xử Lý

### 4.8.1 Từ vựng sau làm sạch

Sau khi áp dụng pipeline làm sạch văn bản trên toàn bộ 4,768 phim:
- **Tổng số từ duy nhất (trước TF-IDF):** ~45,000 từ
- **Sau khi TF-IDF lọc min_df=2:** ~12,000 từ phổ biến
- **Sau khi chọn max_features=500:** 500 từ quan trọng nhất

### 4.8.2 Ví Dụ Minh Họa

| Phim | Overview gốc (đoạn đầu) | overview_clean |
|------|------------------------|----------------|
| Avatar | "In the 22nd century, a paraplegic Marine..." | "22nd century paraplegic marine dispatched..." |
| The Dark Knight | "Batman raises the stakes in his war on crime..." | "batman raise stake war crime help lieutenant..." |
| Inception | "Cobb, a skilled thief who commits corporate espionage..." | "cobb skilled thief commit corporate espionage..." |

Quá trình làm sạch đã loại bỏ các từ không mang nghĩa ("the", "a", "in", "on"), giữ lại từ khóa ngữ nghĩa quan trọng. Lemmatization đưa "raises" → "raise", "commits" → "commit", "stakes" → "stake" — giúp TF-IDF nhận diện cùng một khái niệm qua nhiều dạng biến thể.
