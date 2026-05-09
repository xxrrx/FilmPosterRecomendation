# Module 1: Data Pipeline - Tong Ket Thuc Hien

## Trang Thai
**HOAN THANH**

---

## Muc Tieu
Thu thap, lam sach va chuan hoa du lieu phim tu TMDB 5000 Dataset (Kaggle), fetch poster URL tu TMDB API, xuat ra file `data/processed/movies.csv` san sang cho Module 2.

---

## Nhung Gi Da Thuc Hien

### 1.1 - Tai Dataset Tu Kaggle
- Su dung Kaggle CLI voi API key (`thuannguyen04`)
- Tai ve 2 file vao `data/raw/`:
  - `tmdb_5000_movies.csv` — 4803 phim, 20 cot
  - `tmdb_5000_credits.csv` — 4803 dong, 4 cot (movie_id, title, cast, crew)

### 1.2 - Parse JSON Columns
- Parse cot `genres` (JSON string) → `genres_list` (Python list)
- Parse cot `keywords` → `keywords_list`
- Parse cot `cast` → `cast_list` (top 5 dien vien)
- Parse cot `crew` → `director` (lay nguoi co job = 'Director')
- Merge movies + credits theo cot `id`

### 1.3 - Loc Phim
- Bo phim khong co `overview` hoac overview rong
- Bo phim khong co `genres`
- Ket qua: **4771 phim** con lai sau loc

### 1.4 - Fetch Poster URL Tu TMDB API
- **Van de phat hien:** Dataset Kaggle khong co cot `poster_path`
- **Giai phap:** Goi TMDB API (`/3/movie/{id}`) de lay `poster_path` cho tung phim
- Su dung `ThreadPoolExecutor` (10 threads) de fetch song song
- Co he thong **cache JSON** (`data/raw/poster_paths_cache.json`) tu dong luu moi 200 phim — chay lai an toan neu bi ngat
- Build `poster_url = https://image.tmdb.org/t/p/w500/{poster_path}`
- TMDB API Key su dung: `98a5465c967d9163c57eeb9e3978bb92`

### 1.5 - Chuan Hoa Text
- Lowercase toan bo `overview`
- Xoa ky tu dac biet (chi giu chu cai va khoang trang)
- Xoa stopwords tieng Anh (NLTK stopwords)
- Lemmatize tung tu (WordNetLemmatizer)
- Luu vao cot `overview_clean`

### 1.6 - Xuat File movies.csv
- Chon dung 8 cot theo schema
- Xu ly missing values (year=0, rating=0.0, vote_count=0)
- Luu ra `data/processed/movies.csv`

---

## Ket Qua Dau Ra

### File Output
| File | Vi tri | Mo ta |
|------|--------|-------|
| `movies.csv` | `data/processed/` | Dataset chinh da xu ly |
| `poster_paths_cache.json` | `data/raw/` | Cache poster_path tu TMDB API |

### Thong Ke Dataset (movies.csv)
| Chi so | Gia tri |
|--------|---------|
| So phim | **4752** |
| Nam phat hanh | 1916 - 2017 |
| Rating trung binh | 6.13 / 10 |
| Missing values | 0 |

### Schema movies.csv
```
movie_id    : int     — TMDB movie ID
title       : string  — Ten phim
year        : int     — Nam phat hanh
genres      : string  — JSON array cac the loai
overview_clean : string — Mo ta da clean (lowercase, no stopwords, lemmatized)
poster_url  : string  — https://image.tmdb.org/t/p/w500/{poster_path}
rating      : float   — Diem vote_average (0-10)
vote_count  : int     — So luot vote
```

### Top 5 The Loai Pho Bien
| The loai | So phim |
|----------|---------|
| Drama | 2286 |
| Comedy | 1717 |
| Thriller | 1272 |
| Action | 1152 |
| Romance | 892 |

---

## Van De Gap Phai Va Giai Phap

| Van de | Giai phap |
|--------|-----------|
| Dataset Kaggle khong co `poster_path` | Goi TMDB API fetch tung phim, luu cache |
| Fetch ~4800 phim mat nhieu thoi gian | Dung 10 threads song song + cache de resume |
| Cell notebook bi dao nguoc thu tu | Gop 2 cell thanh 1 theo dung thu tu |

---

## Kiem Tra Dau Ra (Test Pass)
- So phim >= 4500: **PASS** (4752 phim)
- Missing values = 0: **PASS**
- poster_url bat dau bang `https://`: **PASS**
- Du 8 cot schema: **PASS**
- 3 poster URL truy cap duoc: **PASS**

---

## San Sang Cho Module 2
File `data/processed/movies.csv` da san sang lam input cho:
- **Module 2:** Doc `overview_clean` + `poster_url` de trich xuat CNN features va TF-IDF features
