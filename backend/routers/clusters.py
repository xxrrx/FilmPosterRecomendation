"""
routers/clusters.py — /api/clusters
"""

import json
from collections import Counter
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database import get_db, Movie
from schemas import ClustersResponse, ClusterStat, ClusterMoviesResponse, MovieBase

router = APIRouter(prefix="/api/clusters", tags=["clusters"])


def _to_movie_base(m: Movie) -> MovieBase:
    return MovieBase(
        movie_id   = m.movie_id,
        title      = m.title,
        year       = m.year or 0,
        genres     = m.genres or "[]",
        poster_url = m.poster_url or "",
        rating     = m.rating or 0.0,
        vote_count = m.vote_count or 0,
        cluster_id = m.cluster_id if m.cluster_id is not None else -1,
    )


@router.get("", response_model=ClustersResponse)
def get_clusters(db: Session = Depends(get_db)):
    """Thong ke tong quat cac cum phim: so luong, top genres, vi tri PCA."""
    movies = db.query(Movie).all()

    # Nhom theo cluster_id
    cluster_map: dict = {}
    for m in movies:
        cid = m.cluster_id if m.cluster_id is not None else -1
        if cid not in cluster_map:
            cluster_map[cid] = {"movies": [], "pca_x": [], "pca_y": []}
        cluster_map[cid]["movies"].append(m)
        if m.pca_x is not None:
            cluster_map[cid]["pca_x"].append(m.pca_x)
        if m.pca_y is not None:
            cluster_map[cid]["pca_y"].append(m.pca_y)

    cluster_stats = []
    for cid, data in sorted(cluster_map.items()):
        cluster_movies = data["movies"]
        # Top genres trong cum
        all_genres = []
        for m in cluster_movies:
            try:
                all_genres.extend(json.loads(m.genres))
            except Exception:
                pass
        top_genres = [g for g, _ in Counter(all_genres).most_common(3)]

        # Trung tam PCA
        pca_x_vals = data["pca_x"]
        pca_y_vals = data["pca_y"]
        cx = sum(pca_x_vals) / len(pca_x_vals) if pca_x_vals else 0.0
        cy = sum(pca_y_vals) / len(pca_y_vals) if pca_y_vals else 0.0

        cluster_stats.append(ClusterStat(
            cluster_id   = cid,
            movie_count  = len(cluster_movies),
            top_genres   = top_genres,
            center_pca_x = round(cx, 4),
            center_pca_y = round(cy, 4),
        ))

    return ClustersResponse(
        total_clusters = len(cluster_stats),
        clusters       = cluster_stats,
    )


@router.get("/{cluster_id}/movies", response_model=ClusterMoviesResponse)
def get_cluster_movies(
    cluster_id: int,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Lay danh sach phim trong 1 cum cu the."""
    movies = (
        db.query(Movie)
        .filter(Movie.cluster_id == cluster_id)
        .order_by(Movie.vote_count.desc())
        .limit(limit)
        .all()
    )
    if not movies:
        raise HTTPException(status_code=404, detail=f"Khong co phim trong cum {cluster_id}")

    return ClusterMoviesResponse(
        cluster_id  = cluster_id,
        movie_count = len(movies),
        movies      = [_to_movie_base(m) for m in movies],
    )
