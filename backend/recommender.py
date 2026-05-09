"""
recommender.py — Logic tinh cosine similarity va goi y phim
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class Recommender:
    def __init__(self, combined_features: np.ndarray, movie_ids: np.ndarray,
                 cnn_features: np.ndarray, tfidf_matrix: np.ndarray,
                 cluster_labels: np.ndarray):
        """
        combined_features: (N, 2548) normalized
        movie_ids:         (N,) mapping index -> movie_id
        cnn_features:      (N, 2048) raw CNN (truoc normalize)
        tfidf_matrix:      (N, 500)  raw TF-IDF (truoc normalize)
        cluster_labels:    (N,) cluster id cua moi phim
        """
        self.combined   = combined_features
        self.movie_ids  = movie_ids
        self.cnn        = cnn_features
        self.tfidf      = tfidf_matrix
        self.clusters   = cluster_labels

        # Index de tra cuu nhanh movie_id -> array index
        self.id_to_idx = {int(mid): i for i, mid in enumerate(movie_ids)}

    def get_recommendations(
        self,
        movie_id: int,
        top_k: int = 10,
        alpha: float = 0.5,
        same_cluster_only: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Tra ve danh sach (movie_id, similarity_score) goi y cho movie_id.

        alpha: trong so cho CNN features (0=chi dung text, 1=chi dung poster)
        same_cluster_only: chi tim trong cung cum K-Means
        """
        if movie_id not in self.id_to_idx:
            return []

        idx = self.id_to_idx[movie_id]

        # Tao weighted feature vector theo alpha
        # combined = [cnn_norm(2048) | tfidf_norm(500)]
        # Lay cnn phan va tfidf phan tu combined
        cnn_part   = self.combined[idx, :2048]
        tfidf_part = self.combined[idx, 2048:]

        seed_vec = np.concatenate([
            alpha * cnn_part,
            (1 - alpha) * tfidf_part
        ]).reshape(1, -1)

        # Xac dinh pool phim de so sanh
        if same_cluster_only:
            cluster_id = self.clusters[idx]
            pool_mask  = self.clusters == cluster_id
            pool_mask[idx] = False  # Loai chinh phim do
        else:
            pool_mask = np.ones(len(self.combined), dtype=bool)
            pool_mask[idx] = False

        pool_indices = np.where(pool_mask)[0]
        if len(pool_indices) == 0:
            return []

        # Tinh weighted combined matrix cho pool
        pool_cnn   = self.combined[pool_indices, :2048]
        pool_tfidf = self.combined[pool_indices, 2048:]
        pool_matrix = np.concatenate([
            alpha * pool_cnn,
            (1 - alpha) * pool_tfidf
        ], axis=1)

        # Cosine similarity
        sims = cosine_similarity(seed_vec, pool_matrix)[0]

        # Lay top_k
        top_local_idx = sims.argsort()[::-1][:top_k]
        results = []
        for local_i in top_local_idx:
            global_i = pool_indices[local_i]
            mid = int(self.movie_ids[global_i])
            score = float(sims[local_i])
            results.append((mid, score))

        return results
