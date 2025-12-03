"""
models.py

FINAL STABLE VERSION (NO FEATURE MISMATCHES)

Handles:
- Training the match + accept ML models
- Loading saved models
- Loading saved vectorizer (NEVER refits during scoring)
- Featurizing training and scoring pairs safely
"""

import os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.featurize import featurize_pairs, fit_vectorizer, VECTORIZER_PATH


# ==========================================================
# MODEL FILE PATHS (ALWAYS IN models/ FOLDER)
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

MODEL_MATCH_PATH = os.path.join(MODELS_DIR, "model_match.pkl")
MODEL_ACCEPT_PATH = os.path.join(MODELS_DIR, "model_accept.pkl")


# ==========================================================
# Load trained models + vectorizer
# ==========================================================
def load_models_and_vectorizer():
    """Loads match model, accept model, and vectorizer."""
    if not os.path.exists(MODEL_MATCH_PATH) or not os.path.exists(MODEL_ACCEPT_PATH):
        raise FileNotFoundError("Models not found. Train models first.")

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Vectorizer missing. Train models first.")

    with open(MODEL_MATCH_PATH, "rb") as f:
        model_match = pickle.load(f)

    with open(MODEL_ACCEPT_PATH, "rb") as f:
        model_accept = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model_match, model_accept, vectorizer


# ==========================================================
# Training Function (REAL DATA MODE)
# ==========================================================
def train_models(past_df, students_df, internships_df, seed=42):
    """
    Train the match & accept models using REAL student + internship data.
    - NO pref_rank used in training (require_pref_rank=False).
    - Vectorizer is fit ONLY on real student skills + real internship req_skills.
    """

    print("Training models (real-data mode)...")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ------------------------------------------------------
    # FIT VECTORIZER ONCE — ONLY REAL DATA (THIS FIXES FEATURE MISMATCH)
    # ------------------------------------------------------
    vectorizer = fit_vectorizer(
        students_df=students_df,
        internships_df=internships_df,
        load=False      # ALWAYS new during training
    )

    # ------------------------------------------------------
    # Validate required columns in past_df
    # ------------------------------------------------------
    required_cols = [
        "skills",
        "req_skills_job",
        "gpa",
        "stipend_internship",
        "reservation",
        "gender",
        "rural",
        "match",
        "accept"
    ]

    for c in required_cols:
        if c not in past_df.columns:
            raise KeyError(f"Missing column '{c}' in past_df for training.")

    # ------------------------------------------------------
    # Featurize training pairs (NO pref_rank)
    # ------------------------------------------------------
    X = featurize_pairs(past_df, vectorizer, require_pref_rank=False)

    y_match = past_df["match"].astype(int).values
    y_accept = past_df["accept"].astype(int).values

    # ------------------------------------------------------
    # Train-test split
    # ------------------------------------------------------
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X, y_match, test_size=0.20, random_state=seed
    )

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y_accept, test_size=0.20, random_state=seed
    )

    # ------------------------------------------------------
    # Model Definitions
    # ------------------------------------------------------
    model_match = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        random_state=seed
    )

    model_accept = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        random_state=seed
    )

    # ------------------------------------------------------
    # Train Match Model
    # ------------------------------------------------------
    print("Training Match model...")
    model_match.fit(X_train1, y_train1)
    test_pred1 = model_match.predict_proba(X_test1)[:, 1]
    auc1 = roc_auc_score(y_test1, test_pred1)
    print(f"Match Model AUC: {auc1}")

    # ------------------------------------------------------
    # Train Accept Model
    # ------------------------------------------------------
    print("Training Accept model...")
    model_accept.fit(X_train2, y_train2)
    test_pred2 = model_accept.predict_proba(X_test2)[:, 1]
    auc2 = roc_auc_score(y_test2, test_pred2)
    print(f"Accept Model AUC: {auc2}")

    # ------------------------------------------------------
    # SAVE MODELS + VECTORIZER
    # ------------------------------------------------------
    with open(MODEL_MATCH_PATH, "wb") as f:
        pickle.dump(model_match, f)

    with open(MODEL_ACCEPT_PATH, "wb") as f:
        pickle.dump(model_accept, f)

    # Vectorizer already saved inside fit_vectorizer()

    print(f"Models + vectorizer saved to {MODELS_DIR}")

    return model_match, model_accept, vectorizer


# ==========================================================
# SCORING FUNCTION (REAL SCORING MODE)
# ==========================================================
def score_all_pairs(pairs_df, model_match, model_accept, vectorizer):
    """
    Uses trained models + saved vectorizer to compute:
        match_prob + accept_prob
    """

    # ALWAYS require_pref_rank=True for scoring
    X = featurize_pairs(pairs_df, vectorizer, require_pref_rank=True)

    pairs_df["match_score"] = model_match.predict_proba(X)[:, 1]
    pairs_df["accept_score"] = model_accept.predict_proba(X)[:, 1]

    return pairs_df
