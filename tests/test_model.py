"""Tests that pin down the leakage fix and the honest model's quality."""

from src.model import (
    leakage_evidence,
    leaky_baseline_report,
    predict_type,
    top_predictive_terms,
    train_content_classifier,
)


def test_leaky_model_is_suspiciously_perfect(catalogue):
    """Documents the original bug: a 100% score is a red flag, not a result."""
    report = leaky_baseline_report(catalogue)
    assert report.accuracy > 0.99


def test_leakage_evidence_shows_a_clean_separation(catalogue):
    table = leakage_evidence(catalogue)
    # Zero-runtime rows are essentially all TV Shows.
    assert table.loc["duration_mins == 0", "Movie"] < 10
    assert table.loc["duration_mins > 0", "TV Show"] == 0


def test_honest_model_beats_the_majority_baseline(catalogue):
    _, report = train_content_classifier(catalogue, run_cv=False)
    assert report.accuracy > report.baseline_accuracy


def test_honest_model_is_not_perfect(catalogue):
    """If this ever passes 95% again, a leaky feature has crept back in."""
    _, report = train_content_classifier(catalogue, run_cv=False)
    assert report.accuracy < 0.95


def test_honest_model_uses_no_leaky_features(catalogue):
    _, report = train_content_classifier(catalogue, run_cv=False)
    for banned in ("duration", "seasons", "runtime", "listed_in", "rating"):
        assert not any(banned in f for f in report.features_used)


def test_both_classes_are_predicted(catalogue):
    """A model that only ever says 'Movie' would still score 70%."""
    _, report = train_content_classifier(catalogue, run_cv=False)
    assert (report.per_class["recall"] > 0.3).all()


def test_predict_type_returns_a_label_and_confidence(catalogue):
    pipeline, _ = train_content_classifier(catalogue, run_cv=False)
    label, confidence = predict_type(pipeline, "An eight episode crime series.")
    assert label in {"Movie", "TV Show"}
    assert 0.5 <= confidence <= 1.0


def test_top_terms_are_interpretable(catalogue):
    pipeline, _ = train_content_classifier(catalogue, run_cv=False)
    terms = top_predictive_terms(pipeline, n=20)
    assert "series" in terms["TV Show indicator"].tolist()
