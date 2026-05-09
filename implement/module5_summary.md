# Module 5: Frontend (React + Vite + Tailwind) - Tong Ket Thuc Hien

## Trang Thai
**HOAN THANH**

---

## Muc Tieu
Giao dien web truc quan, hien thi ket qua tu tat ca cac mo hinh ML.

---

## Tech Stack
- **React 18** + **Vite 5** — component-based, HMR nhanh
- **Tailwind CSS 3** — styling toan bo bang utility classes
- **React Router v6** — client-side routing 3 trang
- **Axios** — HTTP client goi API backend
- **Recharts** — bieu do scatter plot PCA va bar chart cluster

---

## Cau Truc Thu Muc

```
frontend/src/
├── api/
│   └── movieApi.js          — Axios wrapper: 7 ham goi API
├── components/
│   ├── MovieCard.jsx        — Card poster + ten + rating + cluster badge + NB badge
│   ├── MovieGrid.jsx        — Luoi phim responsive + skeleton loading
│   ├── SearchBar.jsx        — Tim kiem realtime (debounce 350ms) + dropdown
│   ├── GenreTag.jsx         — Badge the loai mau sac khac nhau theo the loai
│   ├── WeightSlider.jsx     — Thanh dieu chinh alpha poster/text co gradient
│   ├── ClusterChart.jsx     — PCA 2D scatter plot (recharts), hover highlight cum
│   └── RulesTable.jsx       — Bang association rules voi mau confidence/lift
└── pages/
    ├── HomePage.jsx         — Trang chu: search + grid + filter genre + phan trang
    ├── MovieDetailPage.jsx  — Chi tiet phim + NB badge + slider alpha + top-10 goi y
    └── ExplorePage.jsx      — PCA chart + bar chart + rules table + tab navigation
```

---

## 3 Trang Chinh

### Trang Chu (`/`)
- Hero section voi SearchBar realtime dropdown
- Filter the loai (14 the loai, click de loc)
- Grid 20 phim/trang (responsive 2-5 cot)
- Phan trang co tong so phim
- Link den trang Kham pha

### Trang Chi Tiet (`/movie/:id`)
- Poster + metadata (ten, nam, rating, vote count)
- Badge the loai thuc te
- Badge **"The loai du doan boi AI (Naive Bayes)"** - hien thi ket qua NB
- Badge cum K-Means co link den trang Explore
- WeightSlider dieu chinh alpha → goi API lai tu dong, cap nhat top-10
- Grid top-10 phim goi y voi badge % similarity

### Trang Kham Pha (`/explore`)
- **Tab PCA 2D:** Scatter plot 4752 phim, mau theo cum, hover tooltip, click mo chi tiet
- **Tab So luong cum:** Bar chart 20 cum, click cot → hien danh sach phim trong cum
- **Tab Association Rules:** Bang 12 luat, to mau theo confidence va lift

---

## Tinh Nang Noi Bat

| Tinh nang | Mo ta |
|-----------|-------|
| Skeleton loading | Hien skeleton cards khi dang goi API |
| Search debounce | Chi goi API sau 350ms ngung go |
| Alpha slider realtime | Thay doi alpha → goi recommend API ngay |
| Similarity badge | Hien % cosine similarity tren moi phim goi y |
| Cluster highlight | Hover vao 1 cum tren scatter plot → mo nhat cac cum khac |
| Poster fallback | Anh loi thi hien placeholder tu dong |
| Proxy Vite | `/api` tu dong forward toi `http://127.0.0.1:8000` |

---

## Cach Chay

```bash
cd frontend
npm install
npm run dev
# Mo trinh duyet: http://localhost:5173
```

---

## Checklist Test Thu Cong
- [x] Search "Avatar" -> hien dung phim, click vao trang chi tiet
- [x] Keo slider alpha -> danh sach goi y thay doi
- [x] Trang Explore -> scatter plot render dung
- [x] Click cot bar chart -> hien phim trong cum
- [x] Bang Association Rules hien du cac luat
- [x] Responsive tren mobile (375px)
