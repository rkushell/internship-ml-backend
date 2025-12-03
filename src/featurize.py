"""
featurize.py

Final stable version — prevents TF-IDF mismatch issues.

Responsibilities:
 - Fit or load TF-IDF vectorizer
 - Create sparse ML features for training + scoring
 - Encode:
      * skills text (TF-IDF)
      * req_skills text (TF-IDF)
      * numeric fields
      * categorical fields (reservation, gender)
 - Ensures training & scoring ALWAYS use identical features
"""

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


# ----------------------------------------------
# PATH TO SAVE/LOAD SKILL VECTORIZER
# ----------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "models", "skill_vectorizer.pkl")
VECTORIZER_PATH = os.path.abspath(VECTORIZER_PATH)

# ======================================================================
# VECTORIZE — TRAIN OR LOAD TF-IDF
# ======================================================================
def fit_vectorizer(students_df=None, internships_df=None, load=True):
    """
    Loads or trains the TF-IDF vectorizer.

    ALWAYS RETURNS the vectorizer.

    load=True  -> load vectorizer from disk (recommended for scoring)
    load=False -> fit a new vectorizer and save it
    """

    # -----------------------------
    # Load existing vectorizer
    # -----------------------------
    if load and os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, "rb") as f:
            cv = pickle.load(f)
        return cv

    # -----------------------------
    # If training a new vectorizer
    # -----------------------------
    if students_df is None or internships_df is None:
        raise ValueError("Must provide students_df and internships_df when load=False")

    text_data = (
        students_df["skills"].astype(str).tolist() +
        internships_df["req_skills"].astype(str).tolist()
    )

    cv = TfidfVectorizer(min_df=2, max_features=5000)
    cv.fit(text_data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)

    # Save vectorizer
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(cv, f)

    return cv


# ======================================================================
# FEATURE GENERATION — TRAINING & SCORING
# ======================================================================
def featurize_pairs(df: pd.DataFrame, vectorizer, require_pref_rank=True):
    """
    Converts pair dataframe → ML feature matrix.

    Required columns (training & scoring):
       - skills
       - req_skills_job
       - gpa
       - stipend_internship
       - reservation
       - gender
       - rural
       - pref_rank (only if require_pref_rank=True)

    Returns: scipy sparse CSR matrix
    """

    if vectorizer is None:
        raise ValueError("Vectorizer cannot be None — must load or train first.")

    # -------------------------------------------------------------
    # Required fields
    # -------------------------------------------------------------
    required_cols = [
        "skills", "req_skills_job", "gpa",
        "stipend_internship", "reservation",
        "gender", "rural"
    ]

    if require_pref_rank:
        required_cols.append("pref_rank")

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in featurize_pairs")

    # -------------------------------------------------------------
    # TEXT FIELDS → TF-IDF ENCODING
    # -------------------------------------------------------------
    skills_vec = vectorizer.transform(df["skills"].astype(str).tolist())
    req_vec = vectorizer.transform(df["req_skills_job"].astype(str).tolist())

    # -------------------------------------------------------------
    # NUMERIC FIELDS
    # -------------------------------------------------------------
    gpa = csr_matrix(df["gpa"].astype(float).values.reshape(-1, 1))
    stipend = csr_matrix(df["stipend_internship"].astype(float).values.reshape(-1, 1))
    rural = csr_matrix(df["rural"].astype(int).values.reshape(-1, 1))

    # -------------------------------------------------------------
    # PREFERENCE RANK
    # -------------------------------------------------------------
    if require_pref_rank:
        pref = csr_matrix(df["pref_rank"].astype(int).values.reshape(-1, 1))
    else:
        pref = csr_matrix(np.zeros((len(df), 1)))

    # -------------------------------------------------------------
    # CATEGORICAL — Reservation
    # -------------------------------------------------------------
    reservation_map = {"GEN": 0, "OBC": 1, "SC": 2, "ST": 3}
    res = csr_matrix(
        df["reservation"].map(reservation_map).fillna(0).astype(int).values.reshape(-1, 1)
    )

    # -------------------------------------------------------------
    # CATEGORICAL — Gender
    # -------------------------------------------------------------
    gender_map = {"M": 0, "F": 1, "O": 2}
    gender = csr_matrix(
        df["gender"].map(gender_map).fillna(0).astype(int).values.reshape(-1, 1)
    )

    # -------------------------------------------------------------
    # FINAL FEATURE MATRIX (sparse)
    # -------------------------------------------------------------
    X = hstack([
        skills_vec,        # TF-IDF student skills
        req_vec,           # TF-IDF job skills
        gpa,
        stipend,
        res,
        gender,
        rural,
        pref
    ]).tocsr()

    return X
