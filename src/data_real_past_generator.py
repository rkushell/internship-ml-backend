"""
data_real_past_generator.py

Realistic synthetic historical generator (v2).

Function:
    generate_pseudo_past_data(students_df, internships_df, n_samples=5000, seed=42, save_path=None, weights=None)

Produces a DataFrame with columns required by training:
    - student_id, internship_id
    - skills, req_skills_job
    - gpa, stipend_internship
    - reservation, gender, rural
    - match (0/1), accept (0/1)

Design notes:
 - Uses a weighted logistic model to compute P(match) from:
     skill_overlap, GPA, internship_attractiveness (stipend/tier), reservation/gender/rural minor effects
 - Uses another model to compute P(accept) conditional on match, influenced by:
     pref_rank (if available), stipend attractiveness, internship tier, location type, random noise
 - Tunable 'weights' parameter lets you change relative importance of factors.
"""

import os
import random
import numpy as np
import pandas as pd
from math import exp


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_series(s):
    s = np.array(s, dtype=float)
    if s.max() == s.min():
        return np.zeros_like(s)
    return (s - s.min()) / (s.max() - s.min())


def generate_pseudo_past_data(
    students_df,
    internships_df,
    n_samples=8000,
    seed=42,
    save_path=None,
    weights=None
):
    """
    Generate a realistic pseudo historical dataset for training.

    Parameters
    ----------
    students_df : pd.DataFrame
        Must contain columns: student_id, skills, gpa, reservation, gender, rural, pref_1..pref_6 (optional)
    internships_df : pd.DataFrame
        Must contain columns: internship_id, req_skills, stipend, tier, location_type (optional)
    n_samples : int
        Number of (student, internship) pairs to sample as historical rows
    seed : int
        Random seed for reproducibility
    save_path : str or None
        If provided, save the resulting CSV to this path
    weights : dict or None
        Tunable weights (recommended keys shown below)

    Returns
    -------
    pd.DataFrame
        Columns: student_id, internship_id, skills, req_skills_job, gpa,
                 stipend_internship, reservation, gender, rural, match, accept
    """

    rand = random.Random(seed)
    np.random.seed(seed)

    # Default weights if not provided
    if weights is None:
        weights = {
            # Match model weights (linear combination before sigmoid)
            "w_skill": 2.0,        # skill overlap importance
            "w_gpa": 0.8,          # gpa importance
            "w_stipend": 0.6,      # stipend/tier importance for match
            "w_tier": 0.6,
            "w_bias": -2.0,        # base bias (controls base match rate)
            # Accept model weights (after match)
            "a_pref": 1.5,         # preference rank weight (higher -> more likely)
            "a_stipend": 0.9,      # stipend attractiveness
            "a_tier": 0.6,
            "a_location_remote_bonus": 0.15,  # remote internships slightly more likely to be accepted
            "a_noise": 0.3,        # random noise added to acceptance logit
            # small demographic nudges (kept small)
            "w_reservation_bias": 0.05,
            "w_gender_bias": 0.03,
            "w_rural_bias": -0.02
        }

    # Helper: extract skill sets
    def _tokens(s):
        if pd.isna(s):
            return set()
        return set(str(s).lower().replace(";", " ").replace(",", " ").split())

    # Precompute skill text lists
    students_skills = students_df["skills"].fillna("").astype(str).apply(_tokens).tolist()
    internships_skills = internships_df["req_skills"].fillna("").astype(str).apply(_tokens).tolist()

    student_id_list = students_df["student_id"].tolist()
    internship_id_list = internships_df["internship_id"].tolist()

    # Precompute numeric attractiveness signals for internships
    stipends = internships_df.get("stipend", pd.Series([0]*len(internships_df))).astype(float).values
    stipend_norm = _normalize_series(stipends)

    # Tier mapping (if tier exists)
    tier_map = {}
    if "tier" in internships_df.columns:
        # map Tier1->2, Tier2->1, Tier3->0 as attractiveness
        def _tier_score(t):
            t = str(t).lower() if not pd.isna(t) else ""
            if "tier1" in t or "1" == t:
                return 2.0
            if "tier2" in t or "2" == t:
                return 1.0
            return 0.0
        tier_scores = internships_df["tier"].fillna("").astype(str).apply(_tier_score).values
        tier_norm = _normalize_series(tier_scores)
    else:
        tier_norm = np.zeros(len(internships_df))

    # location score: remote/hybrid/office/factory
    location_scores = []
    for loc in internships_df.get("location_type", pd.Series([""]*len(internships_df))):
        s = str(loc).lower()
        if "remote" in s:
            location_scores.append(1.0)
        elif "hybrid" in s:
            location_scores.append(0.7)
        elif "office" in s:
            location_scores.append(0.4)
        elif "factory" in s:
            location_scores.append(0.2)
        else:
            location_scores.append(0.3)
    loc_norm = _normalize_series(location_scores)

    # Student GPA normalized
    gpas = students_df.get("gpa", pd.Series([0]*len(students_df))).astype(float).values
    gpa_norm = _normalize_series(gpas)
    # reservation/gender/rural arrays
    reservations = students_df.get("reservation", pd.Series(["GEN"]*len(students_df))).fillna("GEN").astype(str).tolist()
    genders = students_df.get("gender", pd.Series(["M"]*len(students_df))).fillna("M").astype(str).tolist()
    rurals = students_df.get("rural", pd.Series([0]*len(students_df))).fillna(0).astype(int).tolist()

    rows = []

    # To make better coverage, sample students uniformly and internships using stipend/tier bias
    internship_probs = stipend_norm * 0.6 + tier_norm * 0.4
    if internship_probs.sum() <= 0:
        internship_probs = np.ones_like(internship_probs) / len(internship_probs)
    else:
        internship_probs = internship_probs / internship_probs.sum()

    for _ in range(n_samples):
        # sample student uniformly
        si = rand.randrange(len(student_id_list))
        sj = np.random.choice(len(internship_id_list), p=internship_probs)

        student_id = student_id_list[si]
        internship_id = internship_id_list[sj]

        s_sk = students_skills[si]
        j_sk = internships_skills[sj]
        overlap = len(s_sk.intersection(j_sk))

        # normalized overlap score between 0 and 1; use min(1, overlap/len(j_sk))
        if len(j_sk) > 0:
            overlap_score = overlap / float(max(1, len(j_sk)))
        else:
            overlap_score = 0.0

        # Feature combination for match
        # skill contribution amplified by w_skill and somewhat by gpa
        skill_term = weights["w_skill"] * overlap_score
        gpa_term = weights["w_gpa"] * gpa_norm[si]
        stipend_term = weights["w_stipend"] * stipend_norm[sj]
        tier_term = weights["w_tier"] * tier_norm[sj]

        # tiny demographic nudges (not big)
        res_bias = weights.get("w_reservation_bias", 0.0) * (0 if reservations[si] == "GEN" else 0.02)
        gender_bias = weights.get("w_gender_bias", 0.0) * (0.01 if genders[si] == "F" else 0.0)
        rural_bias = weights.get("w_rural_bias", 0.0) * (1 if rurals[si] == 1 else 0)

        logit_match = weights["w_bias"] + skill_term + gpa_term + stipend_term + tier_term + res_bias + gender_bias + rural_bias

        p_match = float(_sigmoid(logit_match))
        # Clip to [0.01, 0.99]
        p_match = max(0.01, min(0.99, p_match))

        # Sample match
        match = int(rand.random() < p_match)

        # Acceptance probability (only meaningful if match==1)
        # Base accept logit favors higher preference, higher stipend, higher tier, remote/hybrid
        # Determine pref_rank if available in students_df (pref_1..pref_6)
        pref_rank = None
        # Try retrieve pref from original students_df
        try:
            student_row = students_df.iloc[si]
            for r in range(1, 7):
                pref_col = f"pref_{r}"
                if pref_col in students_df.columns and str(student_row.get(pref_col)) == str(internship_id):
                    pref_rank = r
                    break
        except Exception:
            pref_rank = None

        # Map pref_rank to value: lower rank => higher base accept
        if pref_rank is None:
            pref_val = 0.0
        else:
            # pref 1 -> 1.0, pref 2 -> 0.85, pref 3 -> 0.7,... pref6->0.25
            pref_scores_map = {1:1.0, 2:0.85, 3:0.7, 4:0.55, 5:0.40, 6:0.25}
            pref_val = pref_scores_map.get(pref_rank, 0.0)

        # combine to acceptance logit
        a_pref_term = weights["a_pref"] * pref_val
        a_stipend_term = weights["a_stipend"] * stipend_norm[sj]
        a_tier_term = weights["a_tier"] * tier_norm[sj]
        a_loc_term = weights.get("a_location_remote_bonus", 0.0) * loc_norm[sj]
        a_noise = np.random.normal(0, weights.get("a_noise", 0.3))

        # base acceptance bias
        base_accept_bias = -1.5  # baseline low accept unless conditions good

        logit_accept = base_accept_bias + a_pref_term + a_stipend_term + a_tier_term + a_loc_term + a_noise

        # Demographic acceptance nudges (very small)
        if reservations[si] in ("SC", "ST"):
            logit_accept += 0.03
        if genders[si] == "F":
            logit_accept += 0.02
        if rurals[si] == 1:
            logit_accept -= 0.01

        p_accept = float(_sigmoid(logit_accept))
        p_accept = max(0.001, min(0.999, p_accept))

        # Final accept is only allowed if match==1; however we can model low-prob accept even if match=0
        if match:
            accept = int(rand.random() < p_accept)
        else:
            # small chance to accept even if not matched (maybe student took unrelated internship)
            accept = int(rand.random() < (0.02 * p_accept))

        row = {
            "student_id": student_id,
            "internship_id": internship_id,
            "skills": " ".join(sorted(list(s_sk))),
            "req_skills_job": " ".join(sorted(list(j_sk))),
            "gpa": float(gpas[si]) if len(gpas)>0 else 0.0,
            "stipend_internship": float(stipends[sj]) if len(stipends)>0 else 0.0,
            "reservation": reservations[si],
            "gender": genders[si],
            "rural": int(rurals[si]),
            "match": int(match),
            "accept": int(accept)
        }

        rows.append(row)

    past_df = pd.DataFrame(rows)

    # Optionally save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        past_df.to_csv(save_path, index=False)

    return past_df
