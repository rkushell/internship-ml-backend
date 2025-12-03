"""
ranklist_builder.py

Builds per-internship ranked student lists using:
- Match probability
- Acceptance probability
- Preference rank score
- Boosting (reservation, gender, rural)

This file prepares the final per-internship sorted lists that the
allocation engine uses.
"""

import pandas as pd

# ---------------------------------------------------------
# Boosting Parameters (tune as needed)
# ---------------------------------------------------------

RESERVATION_BOOST = {
    "GEN": 0.00,
    "OBC": 0.03,
    "SC": 0.05,
    "ST": 0.07
}

RURAL_BOOST = 0.03
FEMALE_BOOST = 0.03

# Soft preference scaling
PREF_SCORES = {
    1: 1.00,
    2: 0.85,
    3: 0.70,
    4: 0.55,
    5: 0.40,
    6: 0.25,
    7: 0.20,   # not in top-6
}


# ---------------------------------------------------------
# Build Ranklists
# ---------------------------------------------------------
def build_ranklists(scored_pairs_df: pd.DataFrame, internships_df: pd.DataFrame):
    """
    Input:
        scored_pairs_df → DataFrame containing:
            student_id, internship_id, match_score, accept_score, pref_rank
            reservation, gender, rural

    Output:
        dict: { internship_id : [ { student info + score }, ... ] }
    """

    required = [
        "student_id",
        "internship_id",
        "match_score",
        "accept_score",
        "pref_rank",
        "reservation",
        "gender",
        "rural"
    ]

    for c in required:
        if c not in scored_pairs_df.columns:
            raise KeyError(f"ranklist_builder missing required column '{c}'")

    ranklists = {}

    # group by internship
    for iid, subdf in scored_pairs_df.groupby("internship_id"):

        df = subdf.copy()

        # -----------------------------------------------------
        # Apply boosts
        # -----------------------------------------------------

        # Reservation boost
        df["reserv_boost"] = df["reservation"].map(RESERVATION_BOOST).fillna(0)

        # Gender boost (female)
        df["gender_boost"] = df["gender"].apply(lambda g: FEMALE_BOOST if g == "F" else 0.0)

        # Rural boost
        df["rural_boost"] = df["rural"].apply(lambda r: RURAL_BOOST if int(r) == 1 else 0.0)

        # Preference score
        df["pref_score"] = df["pref_rank"].apply(lambda r: PREF_SCORES.get(int(r), 0.20))

        # -----------------------------------------------------
        # TOTAL FINAL SCORE
        # -----------------------------------------------------
        # If boosted_score exists (injected earlier), use it instead
        if "boosted_score" in df.columns:
          df["final_score"] = df["boosted_score"] * df["pref_score"]
        else:
          df["final_score"] = (
            df["match_score"] * df["accept_score"] * df["pref_score"]
            + df["reserv_boost"]
            + df["gender_boost"]
            + df["rural_boost"]
          )


        # Sort highest score first
        df = df.sort_values(by="final_score", ascending=False)

        # Convert each row to dict for allocator
        ranklists[iid] = df[[
            "student_id",
            "final_score",
            "reservation",
            "gender",
            "rural",
            "pref_rank",
            "match_score",
            "accept_score",
        ]].to_dict(orient="records")

    print(f"Ranklists built for {len(ranklists)} internships.")
    return ranklists
