"""
routers/movies.py — /api/movies endpoints
"""

import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database import get_db, Movie
from schemas import MovieBase, MovieDetail, MovieListResponse

router = APIRouter(prefix="/api/movies", tags=["movies"])


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


@router.get("/search", response_model=MovieListResponse)
def search_movies(
    q: str = Query(..., min_length=1, description="Ten phim can tim"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Tim kiem phim theo ten (case-insensitive)."""
    results = (
        db.query(Movie)
        .filter(Movie.title.ilike(f"%{q}%"))
        .limit(limit)
        .all()
    )
    return MovieListResponse(
        total  = len(results),
        page   = 1,
        limit  = limit,
        movies = [_to_movie_base(m) for m in results],
    )


@router.get("", response_model=MovieListResponse)
def list_movies(
    page:  int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    genre: Optional[str] = Query(None, description="Loc theo the loai"),
    db: Session = Depends(get_db),
):
    """Danh sach phim phan trang, co the loc theo the loai."""
    query = db.query(Movie)

    if genre:
        # genres luu duoi dang JSON string, dung LIKE de loc
        query = query.filter(Movie.genres.ilike(f'%"{genre}"%'))

    total  = query.count()
    offset = (page - 1) * limit
    movies = query.order_by(Movie.vote_count.desc()).offset(offset).limit(limit).all()

    return MovieListResponse(
        total  = total,
        page   = page,
        limit  = limit,
        movies = [_to_movie_base(m) for m in movies],
    )


@router.get("/{movie_id}", response_model=MovieDetail)
def get_movie(movie_id: int, db: Session = Depends(get_db)):
    """Chi tiet 1 phim: metadata + cluster + NB predicted genres."""
    movie = db.query(Movie).filter(Movie.movie_id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail=f"Khong tim thay phim id={movie_id}")

    return MovieDetail(
        movie_id         = movie.movie_id,
        title            = movie.title,
        year             = movie.year or 0,
        genres           = movie.genres or "[]",
        overview         = movie.overview or "",
        poster_url       = movie.poster_url or "",
        rating           = movie.rating or 0.0,
        vote_count       = movie.vote_count or 0,
        cluster_id       = movie.cluster_id if movie.cluster_id is not None else -1,
        pca_x            = movie.pca_x,
        pca_y            = movie.pca_y,
        predicted_genres = movie.predicted_genres,
    )
