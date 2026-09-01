# Netflix Deep Dive

An interactive Streamlit application over the Netflix catalogue (8,807 titles, 2021
snapshot) joined to an IMDb export. It does three things: explores the catalogue,
recommends titles using content-based similarity, and classifies a synopsis as a
Movie or a TV Show using a **leakage-free** text model.

The headline engineering story of this project is the classifier. The first version
scored a perfect **100% accuracy** — which was not a result, it was a bug. Diagnosing
and fixing that target leakage is documented in full below.

---

## Table of contents

1. [How to run it locally](#how-to-run-it-locally)
2. [What the app does](#what-the-app-does)
3. [Project structure](#project-structure)
4. [The data](#the-data)
5. [The target leakage bug (the important part)](#the-target-leakage-bug-the-important-part)
6. [The model](#the-model)
7. [The recommender](#the-recommender)
8. [Every bug that was fixed](#every-bug-that-was-fixed)
9. [Testing](#testing)
10. [Known limitations](#known-limitations)
11. [Interview prep: questions and answers](#interview-prep-questions-and-answers)
12. [Future work](#future-work)

---

## How to run it locally

**Prerequisites:** Python 3.9 or newer.

```bash
git clone <your-repo-url>
cd "Netflix Deep Dive"
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

On Windows use `venv\Scripts\activate` instead.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Make sure `netflix_titles.csv` and `netflix_imdb.csv` are in the project root — the
app checks for them on startup and shows a clear error if either is missing.

Launch the app:

```bash
streamlit run netflix_app.py
```

It opens at <http://localhost:8501>. First load takes roughly 10–15 seconds while the
catalogue is joined, the TF-IDF index is built and the classifier is cross-validated.
All three are cached, so every interaction after that is instant.

**To run the tests:**

```bash
pip install -r requirements-dev.txt
```

```bash
python -m pytest tests/ -v
```

33 tests, about 10 seconds.

---

## What the app does

Five tabs, with sidebar filters (type, genre, release year, minimum rating) that apply
to the Explore and Watch Planner tabs.

| Tab | What it does |
|---|---|
| **Explore** | Catalogue metrics, crash-safe title search, top-rated titles with a vote-count floor, and four charts: type split, top genres, releases per year, top producing countries, IMDb score distribution. |
| **Recommendations** | Content-based recommendations two ways: *"more like this title"* and free-text *"describe what you feel like watching"*. |
| **Watch Planner** | Given the hours you have free, ranks everything you could actually finish — films and series together. |
| **Model** | The leakage story with evidence, a side-by-side leaky-vs-honest comparison, full metrics (accuracy, macro F1, ROC-AUC, cross-validation, per-class precision/recall, confusion matrix), the model's most predictive terms, and a live classifier you can type into. |
| **Data Quality** | Join diagnostics, per-column missing-value report, and a plainly stated list of limitations. |

---

## Project structure

```
Netflix Deep Dive/
├── netflix_app.py           # Streamlit UI only — widgets and layout
├── src/
│   ├── config.py            # Paths, watch-time assumptions, model constants
│   ├── data.py              # Loading, cleaning, joining, feature engineering
│   ├── model.py             # Classifier, evaluation, leakage demonstration
│   └── recommender.py       # TF-IDF similarity, filtering, watch planning
├── tests/
│   ├── conftest.py          # Shared fixtures (real catalogue + a 4-row fake)
│   ├── test_data.py         # 10 tests — cleaning and join correctness
│   ├── test_model.py        # 8 tests  — leakage and model quality
│   └── test_recommender.py  # 15 tests — filtering, search, similarity
├── netflix_titles.csv       # Netflix catalogue (8,807 rows)
├── netflix_imdb.csv         # IMDb export (5,283 rows)
├── requirements.txt
└── requirements-dev.txt
```

**Why the split matters.** The original project was a single 312-line script where
data cleaning, model training and UI rendering were interleaved at module level.
Nothing could be tested without launching a browser. Now `netflix_app.py` only renders
widgets, and every piece of logic lives in a plain function in `src/` that takes a
DataFrame and returns a DataFrame. That is the change that made the 33 tests possible.

---

## The data

Two independent datasets with **no shared ID column**:

| File | Rows | Contents |
|---|---|---|
| `netflix_titles.csv` | 8,807 | show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description |
| `netflix_imdb.csv` | 5,283 | title, type, imdb_score, imdb_votes, runtime |

**The join.** Because there is no shared key, they are matched on *normalised title
**and** type*. Three deliberate decisions:

1. **Normalise the key** — lower-case and strip whitespace before matching.
2. **Include `type` in the join.** Matching on title alone lets a film inherit the
   score of a series with the same name.
3. **Deduplicate the IMDb side first.** 46 IMDb titles appear more than once
   (remakes, re-releases). A naive join added 43 phantom rows to the catalogue,
   silently inflating every count in the app. The entry with the most votes is kept.

**Result:** 8,807 rows in, 8,807 rows out, **44.3% (3,905 titles)** matched to an
IMDb score.

**The 56% with no score are kept in the catalogue with a genuinely missing value.**
The original code ran `fillna(0)` on the rating. Zero is not a neutral placeholder for
a 1–10 scale — it is the worst possible score, and it quietly ranked more than half
the catalogue below the single worst-reviewed title on Netflix. Unrated titles are now
excluded from rating-based rankings rather than buried by them.

---

## The target leakage bug (the important part)

### The symptom

```
Test accuracy: 1.0000
```

A perfect score on held-out data is almost never good news. On a real problem it means
the model has been handed the answer.

### The diagnosis

The original model predicted `type` (Movie vs TV Show) from three features:

```python
df["season_count"]  = df["duration"].str.extract(r"(\d+)").astype(float)
df["duration_mins"] = df["duration"].apply(lambda x: float(x.split()[0]) if "min" in x else None)
X = df[["imdb_rating", "duration_mins", "season_count"]]
y = df["type_encoded"]
```

Netflix stores runtime and season count in the **same free-text column**: `"90 min"`
for a film, `"2 Seasons"` for a series. So:

- `duration_mins` is populated **only for films** and `None` (then filled with 0) for
  every series.
- After `fillna(0)`, `duration_mins == 0` **is** the label.

The cross-tab makes it unambiguous:

|  | Movie | TV Show |
|---|---|---|
| `duration_mins == 0` | 3 | 2,676 |
| `duration_mins > 0` | 6,128 | 0 |

The separation is perfect. (The 3 films on the wrong side are rows with a missing
duration — the only reason this isn't a literal 100/0 split.) Logistic regression did
not learn anything about films or series; it learned `if duration_mins > 0: Movie`.

There was a **second, independent leak** waiting behind it. The `listed_in` genre
column uses labels like `"TV Dramas"`, `"TV Comedies"`, `"International TV Shows"` —
the literal string `"TV"` appears in the genre of **96% of series and 0% of films**.
Feeding genres to the model scored 99.8%: leakage wearing a different hat. A third,
weaker leak sits in the `rating` certificate column (99.7% of series carry a `TV-`
prefix).

### The fix

Every column derived from `duration`, plus `listed_in` and `rating`, is banned from the
model. They are listed explicitly in `src/config.py` as `LEAKY_COLUMNS`, and
`tests/test_model.py` asserts none of them reach the classifier.

The model was rebuilt on text the label does not hide inside: **description, cast,
director and country**.

### The result

| | Original (leaky) | Rebuilt (honest) |
|---|---|---|
| Test accuracy | **100.0%** | **78.0%** |
| Macro F1 | 1.000 | 0.739 |
| 5-fold CV macro F1 | 0.999 | 0.749 ± 0.016 |
| Features | imdb_rating, duration_mins, season_count | description, cast, director, country |
| Actually learned anything? | No | Yes |

78% looks worse than 100%. It is worth far more: the majority-class baseline
("always guess Movie") is 69.6%, so the honest model beats it by **8.4 points** on a
problem where it can only read a plot synopsis. Both models are still shown side by
side in the app, because the comparison *is* the lesson.

---

## The model

**Task.** Binary classification: is this title a Movie or a TV Show, judging only from
its free-text metadata?

**Pipeline** (`src/model.py`):

```
TfidfVectorizer(max_features=20_000, ngram_range=(1,2),
                stop_words="english", sublinear_tf=True, min_df=2)
        ↓
LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced")
```

**Design choices and why:**

- **TF-IDF over raw counts** — down-weights words common to every synopsis.
- **Bigrams** — `"documentary series"` means something the two words separately do not.
- **`sublinear_tf=True`** — a word appearing 10 times is not 10× as informative.
- **`min_df=2`** — a term appearing once cannot generalise; dropping these cuts the
  vocabulary sharply with no loss.
- **`class_weight="balanced"`** — the catalogue is 70% films. Without this the model
  is rewarded for under-predicting series.
- **Logistic regression over a black box** — the coefficients are directly readable,
  which is how the app can show *what the model learned* rather than just its score.

**Results** (stratified 80/20 split — 7,045 train / 1,762 test):

| Metric | Value |
|---|---|
| Accuracy | 78.0% |
| Majority baseline | 69.6% |
| Macro F1 | 0.739 |
| ROC-AUC | 0.837 |
| 5-fold CV macro F1 | 0.749 ± 0.016 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Movie | 0.839 | 0.847 | 0.843 | 1,227 |
| TV Show | 0.641 | 0.628 | 0.635 | 535 |

Confusion matrix:

|  | Predicted Movie | Predicted TV Show |
|---|---|---|
| **Actual Movie** | 1,039 | 188 |
| **Actual TV Show** | 199 | 336 |

**Sanity check — what it learned.** The strongest coefficients are interpretable:
`series`, `docuseries`, `documentary series` push towards TV Show; `documentary`,
`film` push towards Movie. That is real signal from the synopsis, not an artefact.

**Why the low CV standard deviation matters.** ±0.016 across five folds means the
78% is not a lucky split. A single train/test number can be luck; cross-validation is
the evidence that it isn't.

**Why TV Shows are the harder class.** They are only 30% of the data, and a synopsis
for a series often reads exactly like one for a film — both describe a premise, not a
format. The model is genuinely uncertain, and that uncertainty is honest.

---

## The recommender

The original app was called a "Smart Recommender System" but never recommended
anything — it filtered by genre and sorted by rating. That is a leaderboard, not a
recommender.

**What it does now** (`src/recommender.py`): builds a TF-IDF vector for every title
over its genres, synopsis, cast, director and country, and ranks candidates by
**cosine similarity** to a query.

Two query modes:
- **By title** — "more like *Breaking Bad*". Returns *Better Call Saul* as the top
  hit, which is the sanity check the test suite pins.
- **By free text** — "a spanish heist thriller with a mastermind" surfaces
  *Money Heist* and *La casa de papel*.

**A memory decision worth explaining.** The textbook approach precomputes the full
similarity matrix. Here that is 8,807 × 8,807 float64 ≈ **310 MB**, which would make
the app unusable on a free hosting tier. Instead the sparse TF-IDF matrix is built once
and each query is a single row-against-matrix dot product — the same answers, a
fraction of the memory.

**Why content-based and not collaborative filtering?** Collaborative filtering needs
user–item interaction data. This dataset has none: no users, no ratings-per-user, no
watch history. Content-based is the only honest option here, and it has the side
benefit of no cold-start problem — a brand-new title is recommendable the moment its
synopsis exists.

---

## Every bug that was fixed

| # | Bug | Why it mattered | Fix |
|---|---|---|---|
| 1 | **Target leakage** — model trained on features derived from the label | 100% accuracy that meant nothing | Banned all duration/genre/certificate-derived features; rebuilt on synopsis text |
| 2 | **`duration` parsed with one regex** — `r"(\d+)"` on `"90 min"` returned 90 | Every 90-minute film had a "season count" of 90 | Two separate patterns: `(\d+)\s*min` and `(\d+)\s*Season` |
| 3 | **`estimated_total_time = season_count * 5`** | Applied to films, this said a 90-minute film takes 450 hours | Films use real runtime; series use seasons × episodes × episode length, with the assumption stated in the UI |
| 4 | **`imdb_rating.fillna(0)`** | Ranked 56% of the catalogue below the worst-reviewed title on Netflix | Missing stays missing; unrated titles excluded from rating rankings |
| 5 | **Join on title alone, no dedup** | Added 43 phantom rows; a film could inherit a series' score | Join on normalised title **and** type, after deduplicating IMDb by highest votes |
| 6 | **`str.contains(user_input)` in regex mode** | Typing `(` in the search box crashed the app with an unbalanced-parenthesis error | `regex=False` on all user-supplied search; a test covers `(`, `[`, `*`, `a+b` |
| 7 | **`str.contains(genre)` in regex mode** | `"Children & Family Movies"` was matched as a regex | `regex=False` on genre filtering too |
| 8 | **Hard `if hours <= 4` branch in the planner** | Ask for 5 hours and films became invisible; ask for 2 and series did | Both formats ranked together against one budget, with a format selector |
| 9 | **Model retrained on every widget interaction** | Every slider move retrained the classifier | `@st.cache_resource` for the model and recommender, `@st.cache_data` for the catalogue |
| 10 | **`model.predict([[...]])` on a raw list** | sklearn feature-name warnings on every prediction | Predictions go through the fitted pipeline |
| 11 | **No handling of missing CSVs** | A raw `FileNotFoundError` traceback in the browser | Explicit check with a readable message, then `st.stop()` |
| 12 | **Top-rated ranking ignored vote counts** | A 9.5 from twelve voters outranked *Breaking Bad* | Configurable minimum-vote floor, default 5,000 |
| 13 | **Everything at module level, untestable** | No test could run without a browser | Logic extracted into `src/`; 33 tests |
| 14 | **Unused `model.pkl` in the repo** | A stale artefact nothing loaded | The model is retrained from source; `*.pkl` is gitignored |
| 15 | **Emoji-heavy UI** | Read as decoration rather than a tool | Removed throughout; hierarchy carried by headings, tabs, metrics and tables |

---

## Testing

33 tests in three files, runnable with `python -m pytest tests/ -v`.

Two fixtures: the **real catalogue** (session-scoped, loaded once) and a
**hand-built 4-row catalogue** where every expected value is known by hand — including
a deliberate duplicate title across both types, to prove the join cannot fan out or
cross-match.

Most tests exist to pin a bug that was actually fixed, so a regression fails the suite:

```python
def test_honest_model_is_not_perfect(catalogue):
    """If this ever passes 95% again, a leaky feature has crept back in."""
    _, report = train_content_classifier(catalogue, run_cv=False)
    assert report.accuracy < 0.95
```

That is the single most useful test in the project: it turns "the model is suspiciously
good" from something you have to notice into something CI catches.

---

## Known limitations

Stated plainly, and surfaced in the app's Data Quality tab rather than hidden:

- **56% of titles have no IMDb score.** Title-based matching after normalisation is
  exact, so regional variants and punctuation differences do not match. Fuzzy matching
  would raise coverage at the cost of false pairs.
- **Series watch time is an estimate.** The dataset has no episode counts, so a fixed
  assumption (8 episodes per season, IMDb per-episode runtime where available,
  otherwise 45 minutes) stands in. Film runtimes are exact.
- **The catalogue is a 2021 snapshot.** It does not reflect the current Netflix library.
- **The classifier tops out around 78%.** Descriptions genuinely do not always reveal
  format. This is a ceiling of the data, not a bug.
- **Recommendations are content-based only.** No user data exists in this dataset, so
  no personalisation or collaborative signal is possible.

---

## Interview prep: questions and answers

**"Walk me through this project."**
An interactive Streamlit app over 8,807 Netflix titles joined to IMDb. Three things:
catalogue exploration, a content-based recommender, and a text classifier that decides
whether a synopsis describes a film or a series. The most interesting part is that the
first version of that classifier scored 100%, and finding out why is what the project
is really about.

**"Your model gets 78%. Isn't that bad?"**
The baseline for always guessing the majority class is 69.6%, so it beats that by 8.4
points reading nothing but a plot synopsis. The previous version scored 100% because
it was trained on a feature that was the label in disguise. 78% that means something
beats 100% that means nothing.

**"How did you find the leakage?"**
The score itself was the flag — 100% on held-out data is a signal to go looking, not
to celebrate. I traced each feature back to its source and found all three came from
the `duration` column, which stores minutes for films and seasons for series. A
cross-tab confirmed `duration_mins == 0` separated the classes perfectly. Then I
checked the remaining columns for the same problem and found two more leaks: the genre
labels contain "TV" for 96% of series, and the certificate column has a `TV-` prefix
for 99.7% of them.

**"How do you prevent it coming back?"**
A named `LEAKY_COLUMNS` constant, a test asserting no banned feature reaches the
classifier, and a test asserting the accuracy stays *below* 95%. If someone
reintroduces a leak, CI fails.

**"Why logistic regression and not something stronger?"**
Interpretability was worth more than a point or two of accuracy here. Being able to
show that the model keys on "series" and "docuseries" is what proves it learned
something real rather than another artefact — which, given how the project started, is
the thing that needed proving. A gradient-boosted model would be the next step now
that the features are trustworthy.

**"Why content-based recommendation?"**
Collaborative filtering needs user–item interactions and this dataset has no users at
all. Content-based is the only honest option, and it has no cold-start problem.

**"What would you do with more time?"**
Fuzzy title matching to lift the 44% IMDb coverage; sentence embeddings instead of
TF-IDF for the recommender, so "heist" and "robbery" land near each other; and an
offline evaluation of recommendation quality, which right now is only spot-checked.

**"What's the weakest part of this project?"**
The recommender has no quantitative evaluation. I sanity-check it (Breaking Bad returns
Better Call Saul) and that's pinned in a test, but "does this feel right" is not a
metric. With no ground-truth relevance labels I'd need to construct a proxy — for
example, holding out one title from a franchise and checking whether the others rank
it highly.

---

## Future work

- Fuzzy or ID-based title matching to raise IMDb coverage above 44%.
- Sentence embeddings (e.g. `sentence-transformers`) in place of TF-IDF, so semantic
  neighbours are found rather than just lexical ones.
- Offline recommender evaluation using a franchise-based proxy for relevance.
- A second, more useful prediction task: "will this title be highly rated?", trained
  only on metadata available before release.
- Deploy to Streamlit Community Cloud.

---

## Data sources

- Netflix Movies and TV Shows catalogue (2021 snapshot), `netflix_titles.csv`
- IMDb scores export, `netflix_imdb.csv`

Both are public Kaggle datasets and are included in the repository.
