"""Netflix Deep Dive - Streamlit front end.

This file is deliberately thin: it renders widgets and calls into
``src/`` for anything that counts as logic. That split is what makes the
data cleaning, the recommender and the classifier unit-testable without
a browser (see ``tests/``).

Run with:  streamlit run netflix_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import data as data_module
from src import model as model_module
from src import recommender as rec_module
from src.config import DEFAULT_EPISODE_MINUTES, EPISODES_PER_SEASON

st.set_page_config(
    page_title="Netflix Deep Dive",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading and joining the catalogue...")
def get_data() -> pd.DataFrame:
    return data_module.load_dataset()


@st.cache_resource(show_spinner="Building the recommender index...")
def get_recommender(_df: pd.DataFrame):
    return rec_module.ContentRecommender(_df)


@st.cache_resource(show_spinner="Training and cross-validating the classifier...")
def get_classifier(_df: pd.DataFrame):
    return model_module.train_content_classifier(_df)


@st.cache_resource(show_spinner="Reproducing the original model...")
def get_leaky_report(_df: pd.DataFrame):
    return model_module.leaky_baseline_report(_df)


try:
    df = get_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


def format_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename columns for display and round the noisy float columns."""
    renames = {
        "title": "Title", "type": "Type", "release_year": "Year",
        "imdb_rating": "IMDb", "imdb_votes": "Votes",
        "estimated_watch_hours": "Est. hours", "listed_in": "Genres",
        "similarity": "Similarity", "country": "Country",
        "runtime_minutes": "Runtime (min)", "seasons": "Seasons",
    }
    out = frame.rename(columns=renames)
    for col in ("Est. hours", "Similarity"):
        if col in out.columns:
            out[col] = out[col].round(2)
    return out.reset_index(drop=True)


def histogram(series: pd.Series, edges, labels) -> pd.Series:
    """Bucket a numeric series into labelled bins.

    ``st.bar_chart`` sorts a string index alphabetically, which would put
    '(10, 20]' before '(2, 3]'. Explicit zero-padded labels keep the bars
    in numeric order.
    """
    counts = pd.cut(series, bins=edges, labels=labels).value_counts()
    return counts.reindex(labels).fillna(0).astype(int)


# --------------------------------------------------------------------- #
# Sidebar filters

st.sidebar.title("Filters")
st.sidebar.caption("Applied to the Explore and Watch Planner tabs.")

content_type = st.sidebar.selectbox("Content type", ["All", "Movie", "TV Show"])
genre = st.sidebar.selectbox("Genre", ["All"] + data_module.genre_list(df))

year_min, year_max = int(df["release_year"].min()), int(df["release_year"].max())
year_range = st.sidebar.slider(
    "Release year", year_min, year_max, (2010, year_max)
)
min_rating = st.sidebar.slider("Minimum IMDb rating", 0.0, 10.0, 0.0, 0.1)
rated_only = st.sidebar.checkbox(
    "Only titles with an IMDb score", value=False,
    help="44% of the catalogue matched an IMDb record. The rest are shown "
         "with a blank score rather than a fake 0.0.",
)

filtered = rec_module.apply_filters(
    df, content_type, genre, year_range, min_rating, rated_only
)
st.sidebar.metric("Titles matching filters", f"{len(filtered):,}")

st.title("Netflix Deep Dive")
st.caption(
    "Catalogue exploration, content-based recommendations and a leakage-free "
    "content-type classifier, built on 8,807 Netflix titles joined to IMDb."
)

tab_explore, tab_recommend, tab_planner, tab_model, tab_quality = st.tabs(
    ["Explore", "Recommendations", "Watch Planner", "Model", "Data Quality"]
)


# --------------------------------------------------------------------- #
# Tab 1 - Explore

with tab_explore:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Titles", f"{len(df):,}")
    c2.metric("Movies", f"{(df['type'] == 'Movie').sum():,}")
    c3.metric("TV Shows", f"{(df['type'] == 'TV Show').sum():,}")
    c4.metric("With IMDb score", f"{df['has_rating'].mean():.0%}")

    st.subheader("Search the catalogue")
    query = st.text_input("Title contains", placeholder="e.g. stranger")
    if query:
        hits = rec_module.search_titles(filtered, query, limit=10)
        if hits.empty:
            st.info("No titles matched. Try a shorter search term or widen the filters.")
        else:
            st.dataframe(
                format_table(hits[rec_module.DISPLAY_COLUMNS]),
                use_container_width=True, hide_index=True,
            )

    st.divider()
    st.subheader("Highest rated in the current selection")
    vote_floor = st.slider(
        "Minimum IMDb vote count", 0, 100_000, 5_000, 1_000,
        help="A 9.5 from twelve voters is noise, not a recommendation.",
    )
    top = rec_module.rank_by_rating(filtered, n=10, min_votes=vote_floor)
    if top.empty:
        st.info("Nothing meets those filters. Try lowering the vote floor.")
    else:
        st.dataframe(
            format_table(top[["title", "type", "release_year", "imdb_rating",
                              "imdb_votes", "listed_in"]]),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.subheader("Catalogue insights")
    left, right = st.columns(2)
    with left:
        st.caption("Titles by type")
        st.bar_chart(filtered["type"].value_counts())
        st.caption("Top 10 genres")
        genres = (
            filtered["listed_in"].str.split(",").explode().str.strip()
            .replace("", pd.NA).dropna().value_counts().head(10)
        )
        st.bar_chart(genres)
    with right:
        st.caption("Titles by release year")
        st.line_chart(filtered["release_year"].value_counts().sort_index())
        st.caption("Top 10 producing countries")
        countries = (
            filtered["country"].str.split(",").explode().str.strip()
            .replace("", pd.NA).dropna().value_counts().head(10)
        )
        st.bar_chart(countries)

    st.caption("Distribution of IMDb scores (matched titles only)")
    scores = filtered.loc[filtered["has_rating"], "imdb_rating"]
    if not scores.empty:
        st.bar_chart(
            histogram(
                scores,
                edges=[0, 4, 5, 6, 7, 8, 9, 10],
                labels=["0-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"],
            )
        )


# --------------------------------------------------------------------- #
# Tab 2 - Recommendations

with tab_recommend:
    st.subheader("Content-based recommendations")
    st.caption(
        "Each title is turned into a TF-IDF vector over its genres, synopsis, "
        "cast, director and country. Similarity is the cosine angle between "
        "two of those vectors, so nothing here depends on other users' history."
    )
    recommender = get_recommender(df)

    mode = st.radio(
        "Recommend by", ["A title I liked", "A description of what I want"],
        horizontal=True,
    )

    if mode == "A title I liked":
        seed = st.selectbox(
            "Pick a title",
            options=sorted(df["title"].dropna().unique()),
            index=None,
            placeholder="Start typing a title...",
        )
        col_a, col_b = st.columns(2)
        n_results = col_a.slider("Number of recommendations", 5, 25, 10, key="n_title")
        same_type = col_b.checkbox("Same content type only", value=False)

        if seed:
            results = recommender.similar_to_title(seed, n_results, same_type)
            source = df[df["title"] == seed].iloc[0]
            st.info(
                f"**{source['title']}** ({source['release_year']}) - "
                f"{source['type']} - {source['listed_in']}"
            )
            if results.empty:
                st.warning("No similar titles found for that entry.")
            else:
                st.dataframe(format_table(results), use_container_width=True,
                             hide_index=True)
                st.caption(
                    "Similarity runs 0 to 1. Values above ~0.15 usually mean a "
                    "shared franchise, creator or a very specific genre overlap."
                )
    else:
        text_query = st.text_area(
            "Describe what you feel like watching",
            placeholder="e.g. a slow-burn crime drama set in a small town",
        )
        n_results = st.slider("Number of recommendations", 5, 25, 10, key="n_text")
        if text_query.strip():
            results = recommender.similar_to_text(text_query, n_results)
            if results.empty:
                st.warning(
                    "No overlap with the catalogue vocabulary. Try more common words."
                )
            else:
                st.dataframe(format_table(results), use_container_width=True,
                             hide_index=True)


# --------------------------------------------------------------------- #
# Tab 3 - Watch Planner

with tab_planner:
    st.subheader("What can I finish in the time I have?")
    st.caption(
        f"Films use their real runtime. Series are estimated as "
        f"seasons x {EPISODES_PER_SEASON} episodes x "
        f"{DEFAULT_EPISODE_MINUTES} minutes, using the IMDb per-episode "
        f"runtime when available. Treat series figures as approximations."
    )

    col_a, col_b, col_c = st.columns(3)
    hours = col_a.slider("Hours available", 0.5, 60.0, 3.0, 0.5)
    format_choice = col_b.selectbox("Format", ["Both", "Movie", "TV Show"])
    planner_min_rating = col_c.slider("Minimum IMDb rating", 0.0, 10.0, 7.0, 0.1,
                                      key="planner_rating")

    plan = rec_module.watch_plan(filtered, hours, format_choice, 15, planner_min_rating)
    if plan.empty:
        st.info(
            "Nothing fits that budget under the current filters. Try more hours, "
            "a lower rating floor, or widen the sidebar filters."
        )
    else:
        st.success(
            f"{len(plan)} titles you could finish in {hours:g} hours "
            f"({format_choice.lower()})."
        )
        st.dataframe(format_table(plan), use_container_width=True, hide_index=True)

    st.divider()
    st.caption("How watch time is distributed across the filtered catalogue")
    hours_series = filtered["estimated_watch_hours"].dropna()
    if not hours_series.empty:
        st.bar_chart(
            histogram(
                hours_series.clip(upper=60),
                edges=[0, 1, 2, 3, 5, 10, 20, 40, 60],
                labels=["00-01 h", "01-02 h", "02-03 h", "03-05 h",
                        "05-10 h", "10-20 h", "20-40 h", "40-60 h"],
            )
        )


# --------------------------------------------------------------------- #
# Tab 4 - Model

with tab_model:
    st.subheader("Predicting whether a title is a Movie or a TV Show")

    leaky = get_leaky_report(df)
    pipeline, honest = get_classifier(df)

    st.markdown("#### Why the first version scored 100%")
    st.write(
        "The original model was trained on `imdb_rating`, `duration_mins` and "
        "`season_count`. All three were derived from the `duration` column, "
        "which stores `90 min` for films and `2 Seasons` for series. "
        "`duration_mins` is therefore zero for every series and non-zero for "
        "every film: the feature *is* the label."
    )
    st.dataframe(model_module.leakage_evidence(df), use_container_width=True)
    st.caption(
        "3 films have a missing duration, which is the only reason this is not "
        "a literal 100/0 split. A single if-statement reproduces the model."
    )

    st.markdown("#### Leaky model vs leakage-free model")
    comparison = pd.DataFrame(
        {
            "Original (leaky)": [
                f"{leaky.accuracy:.1%}", f"{leaky.macro_f1:.3f}",
                f"{leaky.cv_mean:.3f}", ", ".join(leaky.features_used),
            ],
            "Rebuilt (honest)": [
                f"{honest.accuracy:.1%}", f"{honest.macro_f1:.3f}",
                f"{honest.cv_mean:.3f}", ", ".join(honest.features_used),
            ],
        },
        index=["Test accuracy", "Macro F1", "5-fold CV macro F1", "Features"],
    )
    st.dataframe(comparison, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{honest.accuracy:.1%}",
              delta=f"{honest.accuracy - honest.baseline_accuracy:+.1%} vs baseline")
    m2.metric("Macro F1", f"{honest.macro_f1:.3f}")
    m3.metric("ROC-AUC", f"{honest.roc_auc:.3f}")
    m4.metric("CV macro F1", f"{honest.cv_mean:.3f} ± {honest.cv_std:.3f}")
    st.caption(
        f"Baseline (always predict the majority class, Movie): "
        f"{honest.baseline_accuracy:.1%}. Train / test split: "
        f"{honest.n_train:,} / {honest.n_test:,}, stratified."
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Per-class performance**")
        st.dataframe(honest.per_class.round(3), use_container_width=True)
        st.caption(
            "TV Shows are the harder class: they are 30% of the data and their "
            "synopses read much like film synopses."
        )
    with col_right:
        st.markdown("**Confusion matrix**")
        st.dataframe(
            pd.DataFrame(
                honest.confusion,
                index=[f"Actual {c}" for c in model_module.CLASS_NAMES],
                columns=[f"Predicted {c}" for c in model_module.CLASS_NAMES],
            ),
            use_container_width=True,
        )

    st.markdown("**What the model actually learned**")
    st.dataframe(model_module.top_predictive_terms(pipeline, 12),
                 use_container_width=True, hide_index=True)
    st.caption(
        "'series', 'docuseries' and 'season' push towards TV Show; "
        "'documentary' and 'film' push towards Movie. Sensible signal, "
        "learned from the synopsis alone."
    )

    st.divider()
    st.markdown("#### Try the classifier")
    user_text = st.text_area(
        "Paste or write a synopsis",
        value="A chemistry teacher diagnosed with cancer turns to manufacturing "
              "drugs to secure his family's future.",
        height=100,
    )
    if st.button("Classify"):
        if not user_text.strip():
            st.warning("Enter a synopsis first.")
        else:
            label, confidence = model_module.predict_type(pipeline, user_text)
            st.success(f"Predicted: **{label}** (confidence {confidence:.1%})")
            if confidence < 0.6:
                st.caption("Low confidence - this synopsis reads ambiguously.")
            similar = get_recommender(df).similar_to_text(user_text, 5)
            if not similar.empty:
                st.markdown("**Closest titles in the catalogue**")
                st.dataframe(format_table(similar), use_container_width=True,
                             hide_index=True)


# --------------------------------------------------------------------- #
# Tab 5 - Data Quality
# --------------------------------------------------------------------- #
with tab_quality:
    st.subheader("Data quality and join diagnostics")
    st.write(
        "The catalogue and the IMDb export are two independent datasets with "
        "no shared key, so they are joined on normalised title **and** type. "
        "That pairing prevents a film inheriting the score of a series with "
        "the same name."
    )

    q1, q2, q3 = st.columns(3)
    q1.metric("Catalogue rows", f"{len(df):,}")
    q2.metric("Matched to IMDb", f"{df['has_rating'].sum():,}",
              delta=f"{df['has_rating'].mean():.1%} coverage")
    q3.metric("Duplicate rows created by the join", "0")

    st.caption(
        "Joining on title alone produced 43 phantom rows, because 46 IMDb "
        "titles are duplicated. The IMDb side is now deduplicated by keeping "
        "the highest-vote entry per (title, type)."
    )

    st.markdown("#### Missing values by column")
    st.dataframe(data_module.data_quality_report(df), use_container_width=True)

    st.markdown("#### Known limitations")
    st.markdown(
        "- **56% of titles have no IMDb score.** They are kept in the "
        "catalogue with a blank score, never a fake `0.0`, and are excluded "
        "from every rating-based ranking.\n"
        "- **Series watch time is an estimate.** Episode counts are not in "
        "the dataset, so a fixed assumption stands in for them.\n"
        "- **Title matching is exact after normalisation.** Regional title "
        "variants and punctuation differences will not match; fuzzy matching "
        "would raise coverage at the cost of false pairs.\n"
        "- **The catalogue is a 2021 snapshot** and does not reflect the "
        "current Netflix library."
    )

st.divider()
st.caption("Netflix Deep Dive - data: Netflix titles (2021) and IMDb export.")
