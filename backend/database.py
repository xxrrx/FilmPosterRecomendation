"""
database.py — SQLAlchemy models + khoi tao SQLite DB tu movies_valid.csv va rules.csv
"""

import json
import os
import numpy as np
import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, Float, Text, String
)
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "movies.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── ORM Models ───────────────────────────────────────────────────────────────

class Movie(Base):
    __tablename__ = "movies"

    movie_id        = Column(Integer, primary_key=True, index=True)
    title           = Column(String, nullable=False, index=True)
    year            = Column(Integer)
    genres          = Column(Text)          # JSON array string
    overview        = Column(Text)
    poster_url      = Column(Text)
    rating          = Column(Float)
    vote_count      = Column(Integer)
    cluster_id      = Column(Integer, index=True)
    pca_x           = Column(Float)
    pca_y           = Column(Float)
    predicted_genres = Column(Text)         # JSON array string (NB output)


class Rule(Base):
    __tablename__ = "rules"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    antecedent  = Column(Text)
    consequent  = Column(Text)
    support     = Column(Float)
    confidence  = Column(Float)
    lift        = Column(Float)


# ─── DB Dependency ────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Init DB tu CSV ───────────────────────────────────────────────────────────

def init_db(nb_model=None, mlb=None, cnn_features=None):
    """
    Tao bang va import du lieu tu movies_valid.csv, rules.csv.
    Neu truyen nb_model + mlb + cnn_features, se tinh predicted_genres.
    """
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Neu da co du lieu thi bo qua
        if db.query(Movie).count() > 0:
            print(f"[DB] Da co {db.query(Movie).count()} phim trong DB, bo qua init.")
            return

        # Load movies_valid.csv
        csv_path = os.path.join(PROJECT_DIR, "data", "processed", "movies_valid.csv")
        df = pd.read_csv(csv_path)
        print(f"[DB] Dang import {len(df)} phim...")

        # Tinh predicted_genres neu co model
        predicted_genres_list = [None] * len(df)
        if nb_model is not None and mlb is not None and cnn_features is not None:
            print("[DB] Dang tinh predicted_genres bang Naive Bayes...")
            preds = nb_model.predict(cnn_features)
            genres_pred = mlb.inverse_transform(preds)
            predicted_genres_list = [json.dumps(list(g)) for g in genres_pred]

        # Insert movies
        movies_to_insert = []
        for i, row in df.iterrows():
            movie = Movie(
                movie_id    = int(row["movie_id"]),
                title       = str(row["title"]),
                year        = int(row.get("year", 0)),
                genres      = str(row["genres"]),
                overview    = str(row.get("overview_clean", "")),
                poster_url  = str(row["poster_url"]),
                rating      = float(row.get("rating", 0.0)),
                vote_count  = int(row.get("vote_count", 0)),
                cluster_id  = int(row.get("cluster_id", -1)),
                pca_x       = float(row.get("pca_x", 0.0)),
                pca_y       = float(row.get("pca_y", 0.0)),
                predicted_genres = predicted_genres_list[i],
            )
            movies_to_insert.append(movie)

        db.bulk_save_objects(movies_to_insert)
        db.commit()
        print(f"[DB] Da import {len(movies_to_insert)} phim.")

        # Load rules.csv
        rules_path = os.path.join(PROJECT_DIR, "models", "rules.csv")
        if os.path.exists(rules_path):
            df_rules = pd.read_csv(rules_path)
            rules_to_insert = []
            for _, row in df_rules.iterrows():
                rule = Rule(
                    antecedent = str(row["antecedents"]),
                    consequent = str(row["consequents"]),
                    support    = float(row["support"]),
                    confidence = float(row["confidence"]),
                    lift       = float(row["lift"]),
                )
                rules_to_insert.append(rule)
            db.bulk_save_objects(rules_to_insert)
            db.commit()
            print(f"[DB] Da import {len(rules_to_insert)} luat ket hop.")

    finally:
        db.close()
