"""
schemas.py — Pydantic response models cho FastAPI
"""

from typing import List, Optional
from pydantic import BaseModel


class MovieBase(BaseModel):
    movie_id:   int
    title:      str
    year:       int
    genres:     str         # JSON string
    poster_url: str
    rating:     float
    vote_count: int
    cluster_id: int


class MovieDetail(MovieBase):
    overview:         Optional[str] = None
    pca_x:            Optional[float] = None
    pca_y:            Optional[float] = None
    predicted_genres: Optional[str] = None  # JSON string


class MovieListResponse(BaseModel):
    total:   int
    page:    int
    limit:   int
    movies:  List[MovieBase]


class RecommendItem(BaseModel):
    movie_id:   int
    title:      str
    year:       int
    genres:     str
    poster_url: str
    rating:     float
    similarity: float


class RecommendResponse(BaseModel):
    seed_movie_id:  int
    seed_title:     str
    alpha:          float
    recommendations: List[RecommendItem]


class RuleItem(BaseModel):
    id:         int
    antecedent: str
    consequent: str
    support:    float
    confidence: float
    lift:       float


class RulesResponse(BaseModel):
    total: int
    rules: List[RuleItem]


class ClusterStat(BaseModel):
    cluster_id:  int
    movie_count: int
    top_genres:  List[str]
    center_pca_x: float
    center_pca_y: float


class ClustersResponse(BaseModel):
    total_clusters: int
    clusters: List[ClusterStat]


class ClusterMoviesResponse(BaseModel):
    cluster_id:  int
    movie_count: int
    movies:      List[MovieBase]
