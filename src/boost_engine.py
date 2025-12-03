import pandas as pd
import numpy as np

def apply_middle_tier_boost(scored_df,
                            k_window=1.0,
                            max_caste_boost=0.10,
                            max_rural_boost=0.15):

    df = scored_df.copy()

    # Step-1: Base fused score (same logic everywhere)
    df["base_score"] = (
        0.6 * df["match_score"] +
        0.4 * df["accept_score"]
    )

    df["boost_amount"] = 0.0
    df["boosted_score"] = df["base_score"]

    for iid, pool in df.groupby("internship_id"):

        median_val = pool["base_score"].median()
        sigma = pool["base_score"].std()
        sigma = 0.01 if sigma < 0.01 else sigma

        window_radius = sigma * k_window

        for idx, row in pool.iterrows():

            score = row["base_score"]
            is_reserved = row.get("reservation", "GEN") in ["SC", "ST", "OBC"]
            if not is_reserved:
                continue

            is_rural = (row.get("rural", 0) == 1)

            dist = abs(score - median_val)
            if dist >= window_radius:
                continue

            factor = 1 - (dist / window_radius)

            boost = max_caste_boost * factor
            if is_rural:
                boost += max_rural_boost * factor

            df.at[idx, "boost_amount"] = boost

    df["boosted_score"] = (df["base_score"] + df["boost_amount"]).clip(upper=1.0)

    return df
