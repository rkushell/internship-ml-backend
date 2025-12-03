import os
import json
import pandas as pd

from src.utils import ensure_dirs
from src.models import (
    load_models_and_vectorizer,
    score_all_pairs,
    train_models,
)
from src.boost_engine import apply_middle_tier_boost
from src.ranklist_builder import build_ranklists
from src.optionC_allotment import optionC_allotment_simulated_rejection
from src.fairness_report import build_fairness_report
from src.boost_report import build_student_boost_report
from src.data_real_past_generator import generate_pseudo_past_data


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output")
JSON_DIR = os.path.join(ROOT, "json_outputs")
MODELS_DIR = os.path.join(ROOT, "models")

RESULT_PATH = os.path.join(JSON_DIR, "last_results.json")
ROUND_LOG_PATH = os.path.join(JSON_DIR, "round_logs.json")
BOOST_JSON = os.path.join(JSON_DIR, "boost_report.json")
FINAL_ALLOC_JSON = os.path.join(JSON_DIR, "final_allocations.json")
FAIRNESS_JSON = os.path.join(JSON_DIR, "fairness_report.json")


def train_models_service(
    n_samples_past: int = 12000,
    generator_seed: int = 123,
    train_seed: int = 42,
):
    """
    Heavy training step.

    - Loads data/students.csv and data/internships.csv
    - Generates pseudo historical pairs
    - Trains match + accept models
    - Saves models + vectorizer in /models

    You do NOT need to call this frequently.
    """
    ensure_dirs(DATA_DIR, OUTPUT_DIR, JSON_DIR, MODELS_DIR)

    students_path = os.path.join(DATA_DIR, "students.csv")
    internships_path = os.path.join(DATA_DIR, "internships.csv")

    if not os.path.exists(students_path):
        raise FileNotFoundError("students.csv missing in /data")

    if not os.path.exists(internships_path):
        raise FileNotFoundError("internships.csv missing in /data")

    students_df = pd.read_csv(students_path)
    internships_df = pd.read_csv(internships_path)

    # Generate pseudo-past data
    past_pairs_path = os.path.join(DATA_DIR, "past_pairs_gen.csv")
    past_df = generate_pseudo_past_data(
        students_df=students_df,
        internships_df=internships_df,
        n_samples=n_samples_past,
        seed=generator_seed,
        save_path=past_pairs_path,
        weights=None,
    )

    # Train models using real data + generated past_df
    train_models(
        past_df=past_df,
        students_df=students_df,
        internships_df=internships_df,
        seed=train_seed,
    )

    return {
        "message": "Training completed successfully",
        "students": len(students_df),
        "internships": len(internships_df),
        "past_pairs_generated": len(past_df),
        "models_dir": MODELS_DIR,
        "past_pairs_path": past_pairs_path,
    }


def run_allocation_pipeline():
    """
    Fast allocation step:

    - Loads students.csv + internships.csv
    - Loads trained models + vectorizer
    - Scores all student-intern pairs
    - Applies boosting
    - Builds ranklists
    - Runs multi-round allocation
    - Builds fairness + boost reports
    - Writes combined dashboard JSON to last_results.json
    """
    ensure_dirs(DATA_DIR, OUTPUT_DIR, JSON_DIR, MODELS_DIR)

    students_path = os.path.join(DATA_DIR, "students.csv")
    internships_path = os.path.join(DATA_DIR, "internships.csv")

    if not os.path.exists(students_path):
        raise FileNotFoundError("students.csv missing in /data")

    if not os.path.exists(internships_path):
        raise FileNotFoundError("internships.csv missing in /data")

    students_df = pd.read_csv(students_path)
    internships_df = pd.read_csv(internships_path)

    # Load ML models & vectorizer (already trained)
    model_match, model_accept, vectorizer = load_models_and_vectorizer()

    # Build all student-intern pairs
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

    # Preference rank
    def get_pref_rank(row):
        iid = row["internship_id"]
        for r in range(1, 7):
            if row.get(f"pref_{r}") == iid:
                return r
        return 7

    pairs_df["pref_rank"] = pairs_df.apply(get_pref_rank, axis=1)

    # ML scoring
    scored = score_all_pairs(pairs_df, model_match, model_accept, vectorizer)

    # Apply middle-tier boosting
    boosted = apply_middle_tier_boost(scored)

    # Build ranklists
    ranklists = build_ranklists(boosted, internships_df)

    # Run allocation simulation
    final_df, round_logs = optionC_allotment_simulated_rejection(
        ranklists=ranklists,
        internships_df=internships_df,
        out_json_dir=JSON_DIR,
        max_rounds=8,
        seed=123,
    )

    # Fairness report
    fairness_report = build_fairness_report(final_df, students_df, round_logs)

    # Boost report
    boost_report = build_student_boost_report(
        boosted_df=boosted,
        final_alloc_df=final_df,
        out_path=BOOST_JSON,
    )

    # Save outputs as JSON for dashboard & downloads
    with open(FINAL_ALLOC_JSON, "w") as f:
        json.dump(final_df.to_dict(orient="records"), f, indent=2)

    with open(FAIRNESS_JSON, "w") as f:
        json.dump(fairness_report, f, indent=2)

    with open(ROUND_LOG_PATH, "w") as f:
        json.dump(round_logs, f, indent=2)

    with open(BOOST_JSON, "w") as f:
        json.dump(boost_report, f, indent=2)

    # Combined dashboard view
    results = {
        "students": len(students_df),
        "internships": len(internships_df),
        "allocations_count": len(final_df),
        "allocations": final_df.to_dict(orient="records"),
        "fairness": fairness_report,
        "round_logs": round_logs,
        "boost_report": boost_report,
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return {
        "message": "Allocation completed successfully",
        "summary": results,
    }
