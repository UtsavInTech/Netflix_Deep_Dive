"""Content-type classifier, plus a reproducible demonstration of the
target leakage that made the original model score a perfect 100%.

The task: given only the free-text metadata of a title (description,
cast, director, country), decide whether it is a Movie or a TV Show.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, RANDOM_STATE, TEST_SIZE

CLASS_NAMES = ["Movie", "TV Show"]


@dataclass
class ModelReport:
    """Everything the Model tab needs to describe a trained classifier."""

    name: str
    accuracy: float
    macro_f1: float
    roc_auc: float
    baseline_accuracy: float
    cv_mean: float
    cv_std: float
    confusion: np.ndarray
    per_class: pd.DataFrame
    n_train: int
    n_test: int
    features_used: list = field(default_factory=list)


def _label(df: pd.DataFrame) -> pd.Series:
    """1 = TV Show, 0 = Movie."""
    return (df["type"] == "TV Show").astype(int)


def _summarise(name, y_test, y_pred, y_proba, cv_scores, n_train, features):
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "precision": [report[c]["precision"] for c in CLASS_NAMES],
            "recall": [report[c]["recall"] for c in CLASS_NAMES],
            "f1-score": [report[c]["f1-score"] for c in CLASS_NAMES],
            "support": [int(report[c]["support"]) for c in CLASS_NAMES],
        },
        index=CLASS_NAMES,
    )
    return ModelReport(
        name=name,
        accuracy=accuracy_score(y_test, y_pred),
        macro_f1=f1_score(y_test, y_pred, average="macro"),
        roc_auc=roc_auc_score(y_test, y_proba),
        # A model must beat "always guess the majority class" to be useful.
        baseline_accuracy=max(np.mean(y_test == 0), np.mean(y_test == 1)),
        cv_mean=float(np.mean(cv_scores)),
        cv_std=float(np.std(cv_scores)),
        confusion=confusion_matrix(y_test, y_pred),
        per_class=per_class,
        n_train=n_train,
        n_test=len(y_test),
        features_used=features,
    )


# --------------------------------------------------------------------- #
# The honest model
# --------------------------------------------------------------------- #
def build_pipeline() -> Pipeline:
    """TF-IDF over metadata text, then regularised logistic regression.

    ``class_weight='balanced'`` matters here: the catalogue is 70% films,
    so an unweighted model is tempted to under-predict series.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20_000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, C=5.0, class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_content_classifier(df: pd.DataFrame, run_cv: bool = True):
    """Fit the leakage-free classifier and score it on a held-out split."""
    X = df["model_text"]
    y = _label(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Cross-validation on the full dataset guards against a lucky split.
    cv_scores = (
        cross_val_score(build_pipeline(), X, y, cv=CV_FOLDS, scoring="f1_macro")
        if run_cv
        else [f1_score(y_test, y_pred, average="macro")]
    )

    report = _summarise(
        "Honest text classifier", y_test, y_pred, y_proba, cv_scores,
        len(y_train), ["description", "cast", "director", "country"],
    )
    return pipeline, report


def leaky_baseline_report(df: pd.DataFrame) -> ModelReport:
    """Rebuild the original model to show *why* it scored 100%.

    It was trained on the first integer found in ``duration``. For a film
    that is its runtime; for a series it is the season count. The two
    ranges never overlap, so the feature simply *is* the label wearing a
    numeric disguise.
    """
    features = pd.DataFrame(
        {
            "imdb_rating": df["imdb_rating"].fillna(0),
            "duration_mins": df["runtime_minutes"].fillna(0),
            "season_count": df["duration"]
            .fillna("")
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
            .fillna(0),
        }
    )
    y = _label(df)
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(
        LogisticRegression(max_iter=1000), features, y, cv=CV_FOLDS, scoring="f1_macro"
    )
    return _summarise(
        "Original leaky model", y_test, y_pred, y_proba, cv_scores,
        len(y_train), ["imdb_rating", "duration_mins", "season_count"],
    )


def leakage_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab proving a single feature separates the classes perfectly."""
    zero_runtime = df["runtime_minutes"].fillna(0) == 0
    table = pd.crosstab(
        zero_runtime.map({True: "duration_mins == 0", False: "duration_mins > 0"}),
        df["type"],
    )
    table.index.name = "Leaky feature value"
    return table


def top_predictive_terms(pipeline: Pipeline, n: int = 15) -> pd.DataFrame:
    """Terms pushing hardest towards each class - a sanity check that the
    model learned something meaningful rather than an artefact."""
    vectorizer = pipeline.named_steps["tfidf"]
    coefs = pipeline.named_steps["clf"].coef_[0]
    terms = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(coefs)
    return pd.DataFrame(
        {
            "Movie indicator": terms[order[:n]],
            "Movie weight": np.round(coefs[order[:n]], 3),
            "TV Show indicator": terms[order[-n:][::-1]],
            "TV Show weight": np.round(coefs[order[-n:][::-1]], 3),
        }
    )


def predict_type(pipeline: Pipeline, text: str):
    """Classify one free-text synopsis. Returns (label, confidence)."""
    proba = pipeline.predict_proba([text])[0]
    idx = int(np.argmax(proba))
    return CLASS_NAMES[idx], float(proba[idx])
