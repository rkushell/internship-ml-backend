#!/usr/bin/env python3
"""
main.py
Pipeline entrypoint using realistic synthetic past generator v2.
"""

import os
import json
import pandas as pd

from src.utils import ensure_dirs
from src.data_real_past_generator import generate_pseudo_past_data
from src.models import train_models, score_all_pairs
from src.boost_engine import apply_middle_tier_boost
from src.ranklist_builder import build_ranklists
from src.optionC_allotment import optionC_allotment_simulated_rejection
from src.fairness_report import build_fairness_report
from src.boost_report import build_student_boost_report


# Directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
JSON_DIR = os.path.join(ROOT_DIR, "json_outputs")

RANDOM_SEED = 123


def main(
    n_samples_past=12000,
    generator_seed=123,
    generator_weights=None
):
    ensure_dirs(DATA_DIR, MODELS_DIR, OUTPUT_DIR, JSON_DIR)

    print("Loading real datasets...")

    students_path = os.path.join(DATA_DIR, "students.csv")
    internships_path = os.path.join(DATA_DIR, "internships.csv")

    students_df = pd.read_csv(students_path)
    internships_df = pd.read_csv(internships_path)

    # Normalize semicolon separators
    if "skills" in students_df.columns:
        students_df["skills"] = students_df["skills"].astype(str).str.replace(";", " ")
    if "req_skills" in internships_df.columns:
        internships_df["req_skills"] = internships_df["req_skills"].astype(str).str.replace(";", " ")

    if "capacity" not in internships_df.columns:
        raise KeyError("internships.csv must contain a 'capacity' column.")

    print(f"Loaded {len(students_df)} students and {len(internships_df)} internships")

    # ----------------------------------------------------------------------
    # Generate realistic pseudo-past match + accept data
    # ----------------------------------------------------------------------
    print("Generating realistic pseudo historical training pairs (v2)...")

    past_df = generate_pseudo_past_data(
        students_df=students_df,
        internships_df=internships_df,
        n_samples=n_samples_past,
        seed=generator_seed,
        save_path=os.path.join(DATA_DIR, "past_pairs_gen.csv"),
        weights=generator_weights
    )

    print(f"Generated pseudo past pairs saved to {os.path.join(DATA_DIR, 'past_pairs_gen.csv')}")

    # ----------------------------------------------------------------------
    # Train models
    # ----------------------------------------------------------------------
    print("Training models...")

    model_match, model_accept, vectorizer = train_models(
        past_df,
        students_df,
        internships_df,
        seed=RANDOM_SEED
    )

    print(f"Models + vectorizer saved to {MODELS_DIR}")

    # ----------------------------------------------------------------------
    # Build all candidate–internship pairs
    # ----------------------------------------------------------------------
    print("Scoring student-internship pairs...")

    pairs = []
    for _, s in students_df.iterrows():
        for _, j in internships_df.iterrows():
            pairs.append({
                "student_id": s["student_id"],
                "internship_id": j["internship_id"],
                "skills": s.get("skills", ""),
                "req_skills_job": j.get("req_skills", ""),
                "gpa": s.get("gpa", 0.0),
                "stipend_internship": j.get("stipend", 0.0),
                "reservation": s.get("reservation", "GEN"),
                "gender": s.get("gender", "M"),
                "rural": s.get("rural", 0),
                "pref_1": s.get("pref_1"),
                "pref_2": s.get("pref_2"),
                "pref_3": s.get("pref_3"),
                "pref_4": s.get("pref_4"),
                "pref_5": s.get("pref_5"),
                "pref_6": s.get("pref_6"),
            })

    pairs_df = pd.DataFrame(pairs)
    print(f"Total pairs to score: {len(pairs_df)}")

    # ----------------------------------------------------------------------
    # Compute preference rank for each pair
    # ----------------------------------------------------------------------
    print("Assigning preference ranks...")

    def get_pref_rank(row):
        iid = row["internship_id"]
        for r in range(1, 7):
            if row.get(f"pref_{r}") == iid:
                return r
        return 7

    pairs_df["pref_rank"] = pairs_df.apply(get_pref_rank, axis=1)

    # ----------------------------------------------------------------------
    # ML scoring
    # ----------------------------------------------------------------------
    scored_pairs = score_all_pairs(pairs_df, model_match, model_accept, vectorizer)

    # ----------------------------------------------------------------------
    # Apply fairness middle-tier boosting
    # ----------------------------------------------------------------------
    scored_pairs = apply_middle_tier_boost(scored_pairs)

    # Save debug boosted file
    boosted_path = os.path.join(OUTPUT_DIR, "boosted_pairs_debug.csv")
    scored_pairs.to_csv(boosted_path, index=False)
    print(f"Boosted scored pairs saved to: {boosted_path}")

    # ----------------------------------------------------------------------
    # Build ranklists (uses boosted scoring)
    # ----------------------------------------------------------------------
    print("Building ranklists...")
    ranklists = build_ranklists(scored_pairs, internships_df)
    print(f"Built ranklists for {len(ranklists)} internships.")

    # ----------------------------------------------------------------------
    # Run allocation simulation
    # ----------------------------------------------------------------------
    print("Running hybrid simulated allocation...")
    final_df, round_logs = optionC_allotment_simulated_rejection(
        ranklists=ranklists,
        internships_df=internships_df,
        out_json_dir=JSON_DIR,
        max_rounds=8,
        default_accept_prob=0.7,
        seed=RANDOM_SEED
    )

    final_path = os.path.join(OUTPUT_DIR, "final_allocations_real.csv")
    final_df.to_csv(final_path, index=False)

    # ----------------------------------------------------------------------
    # Build student-level boost uplift impact JSON
    # ----------------------------------------------------------------------
    student_boost_path = os.path.join(JSON_DIR, "student_boost_impact.json")
    build_student_boost_report(
        boosted_df=scored_pairs,
        final_alloc_df=final_df,
        out_path=student_boost_path
    )
    print(f"Student-level boost impact saved to: {student_boost_path}")

    # ----------------------------------------------------------------------
    # Build fairness report JSON
    # ----------------------------------------------------------------------
    print("Generating fairness report...")
    fairness_report = build_fairness_report(
        final_alloc_df=final_df,
        students_df=students_df,
        round_logs=round_logs
    )

    fairness_path = os.path.join(JSON_DIR, "final_fairness_report.json")
    with open(fairness_path, "w") as f:
        json.dump(fairness_report, f, indent=2)

    print(f"Final allocations saved to: {final_path}")
    print(f"JSON logs saved in: {JSON_DIR}")
    print("\nPipeline Completed Successfully.")


# Entry point
if __name__ == "__main__":
    main(n_samples_past=12000, generator_seed=RANDOM_SEED)
