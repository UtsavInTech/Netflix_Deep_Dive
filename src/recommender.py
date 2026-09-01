"""Content-based recommendation and watch-time planning.

The original app was called a 'recommender' but only ever sorted the
catalogue by rating. This module adds the missing piece: a TF-IDF vector
space over each title's genres, synopsis, cast, director and country,
with cosine similarity as the notion of 'alike'.

Similarities are computed one query at a time. A full 8,807 x 8,807 dense
matrix would be ~310 MB; a single query row against the sparse matrix is
one cheap dot product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DISPLAY_COLUMNS = ["title", "type", "release_year", "imdb_rating",
                   "estimated_watch_hours", "listed_in"]


class ContentRecommender:
    """Fits once over the catalogue, then answers similarity queries."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=2,
            max_features=50_000, sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.df["content_text"])
        # Title -> row index, lower-cased for forgiving lookups.
        self._index = pd.Series(
            self.df.index, index=self.df["title"].str.strip().str.lower()
        )
        self._index = self._index[~self._index.index.duplicated()]

    # ----------------------------------------------------------------- #
    def has_title(self, title: str) -> bool:
        return str(title).strip().lower() in self._index.index

    def similar_to_title(self, title: str, n: int = 10, same_type_only: bool = False):
        """Titles closest to ``title`` in the TF-IDF space."""
        key = str(title).strip().lower()
        if key not in self._index.index:
            return pd.DataFrame(columns=DISPLAY_COLUMNS + ["similarity"])

        idx = int(self._index.loc[key])
        scores = cosine_similarity(self.matrix[idx], self.matrix).ravel()
        scores[idx] = -1.0  # never recommend the query back to the user

        candidates = self.df.assign(similarity=scores)
        if same_type_only:
            candidates = candidates[candidates["type"] == self.df.at[idx, "type"]]
        return self._top(candidates, n)

    def similar_to_text(self, query: str, n: int = 10):
        """Free-text search: 'a heist thriller set in Spain'."""
        query = str(query).strip()
        if not query:
            return pd.DataFrame(columns=DISPLAY_COLUMNS + ["similarity"])
        vector = self.vectorizer.transform([query])
        scores = cosine_similarity(vector, self.matrix).ravel()
        return self._top(self.df.assign(similarity=scores), n)

    @staticmethod
    def _top(candidates: pd.DataFrame, n: int) -> pd.DataFrame:
        result = candidates[candidates["similarity"] > 0].nlargest(n, "similarity")
        return result[DISPLAY_COLUMNS + ["similarity"]].reset_index(drop=True)


# --------------------------------------------------------------------- #
# Filtering and ranking
# --------------------------------------------------------------------- #
def apply_filters(df, content_type="All", genre="All", year_range=None,
                  min_rating=0.0, rated_only=False):
    """Sidebar filters. ``regex=False`` is deliberate - genre names such as
    'Children & Family Movies' contain characters that break a regex."""
    out = df
    if content_type != "All":
        out = out[out["type"] == content_type]
    if genre != "All":
        out = out[out["listed_in"].str.contains(genre, case=False, na=False, regex=False)]
    if year_range is not None:
        out = out[out["release_year"].between(year_range[0], year_range[1])]
    if rated_only:
        out = out[out["has_rating"]]
    if min_rating > 0:
        out = out[out["imdb_rating"].fillna(-1) >= min_rating]
    return out


def search_titles(df: pd.DataFrame, query: str, limit: int = 10) -> pd.DataFrame:
    """Substring title search. ``regex=False`` stops a stray '(' in the
    search box from raising a regex error and crashing the app."""
    query = str(query).strip()
    if not query:
        return df.head(0)
    hits = df[df["title"].str.contains(query, case=False, na=False, regex=False)]
    return hits.sort_values("imdb_rating", ascending=False, na_position="last").head(limit)


def rank_by_rating(df: pd.DataFrame, n: int = 10, min_votes: int = 0) -> pd.DataFrame:
    """Top titles by IMDb score.

    Unrated titles are excluded rather than treated as 0.0, and an
    optional vote floor keeps a 9.5 from eleven voters out of the chart.
    """
    rated = df[df["has_rating"]]
    if min_votes > 0:
        rated = rated[rated["imdb_votes"].fillna(0) >= min_votes]
    return rated.nlargest(n, "imdb_rating")


def fits_in_time(df: pd.DataFrame, hours_available: float, tolerance: float = 0.0):
    """Everything finishable within the viewer's time budget.

    The original app branched on ``hours <= 4``: ask for five hours and it
    would only ever show series, and ask for two and films were the only
    option. Both formats are now ranked together against one budget.
    """
    budget = hours_available * (1 + tolerance)
    within = df[df["estimated_watch_hours"].notna() & (df["estimated_watch_hours"] <= budget)]
    return within


def watch_plan(df, hours_available, content_type="Both", n=10, min_rating=0.0):
    """Ranked shortlist of titles that fit the available time."""
    candidates = fits_in_time(df, hours_available)
    if content_type != "Both":
        candidates = candidates[candidates["type"] == content_type]
    if min_rating > 0:
        candidates = candidates[candidates["imdb_rating"].fillna(-1) >= min_rating]
    candidates = candidates[candidates["has_rating"]]
    return candidates.nlargest(n, "imdb_rating")[DISPLAY_COLUMNS]
