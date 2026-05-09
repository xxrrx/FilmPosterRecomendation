"""
routers/recommend.py — /api/movies/{id}/recommend
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from database import get_db, Movie
from schemas import RecommendResponse, RecommendItem

router = APIRouter(prefix="/api/movies", tags=["recommend"])


@router.get("/{movie_id}/recommend", response_model=RecommendResponse)
def recommend(
    movie_id: int,
    request:  Request,
    alpha:    float = Query(0.5, ge=0.0, le=1.0,
                            description="Trong so poster (0=chi text, 1=chi poster)"),
    top_k:    int   = Query(10, ge=1, le=50),
    db:       Session = Depends(get_db),
):
    """
    Goi y top-k phim tuong tu dua tren combined features.
    alpha dieu chinh trong so poster vs text.
    """
    # Kiem tra phim ton tai
    seed = db.query(Movie).filter(Movie.movie_id == movie_id).first()
    if not seed:
        raise HTTPException(status_code=404, detail=f"Khong tim thay phim id={movie_id}")

    # Lay recommender tu app.state
    recommender = request.app.state.recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender chua san sang")

    # Lay danh sach (movie_id, score)
    results = recommender.get_recommendations(movie_id, top_k=top_k, alpha=alpha)

    if not results:
        return RecommendResponse(
            seed_movie_id    = movie_id,
            seed_title       = seed.title,
            alpha            = alpha,
            recommendations  = [],
        )

    # Lay thong tin phim tu DB
    rec_ids = [r[0] for r in results]
    score_map = {r[0]: r[1] for r in results}

    movies = db.query(Movie).filter(Movie.movie_id.in_(rec_ids)).all()
    movie_map = {m.movie_id: m for m in movies}

    recommendations = []
    for mid, score in results:
        m = movie_map.get(mid)
        if m:
            recommendations.append(RecommendItem(
                movie_id   = m.movie_id,
                title      = m.title,
                year       = m.year or 0,
                genres     = m.genres or "[]",
                poster_url = m.poster_url or "",
                rating     = m.rating or 0.0,
                similarity = round(score, 4),
            ))

    return RecommendResponse(
        seed_movie_id   = movie_id,
        seed_title      = seed.title,
        alpha           = alpha,
        recommendations = recommendations,
    )
