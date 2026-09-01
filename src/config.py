"""Central configuration: file paths and tunable constants.

Keeping these in one place means the Streamlit app, the tests and any
notebook all agree on where the data lives and what assumptions we make.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NETFLIX_CSV = PROJECT_ROOT / "netflix_titles.csv"
IMDB_CSV = PROJECT_ROOT / "netflix_imdb.csv"

# --- Watch-time assumptions ---------------------------------------------
# The Netflix catalogue gives a season count but never an episode count, so
# total watch time for a series has to be estimated. These numbers are
# assumptions, not facts, and are surfaced in the UI so nobody mistakes
# them for measurements.
EPISODES_PER_SEASON = 8
DEFAULT_EPISODE_MINUTES = 45

# --- Modelling ----------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Columns that must never reach the classifier: each one encodes the
# target ("Movie" vs "TV Show") almost perfectly. See README, "Target
# leakage" section.
LEAKY_COLUMNS = ("duration", "runtime_minutes", "seasons", "listed_in", "rating")
