import os
import json
import pandas as pd

from src.models import load_models_and_vectorizer, score_all_pairs
from src.boost_engine import apply_middle_tier_boost
from src.ranklist_builder import build_ranklists
from src.optionC_allotment import optionC_allotment_simulated_rejection
from src.fairness_report import build_fairness_report
from src.boost_report import build_student_boost_report

DATA_DIR = "data"
OUTPUT_DIR = "output"
JSON_DIR = "json_outputs"
MODELS_DIR = "models"

LAST_RESULTS = os.path.join(JSON_DIR, "last_results.json")
FINAL_ALLOC_CSV = os.path.join(OUTPUT_DIR, "final_allocations.csv")
FINAL_ALLOC_JSON = os.path.join(JSON_DIR, "final_allocations.json")
FAIRNESS_JSON = os.path.join(JSON_DIR, "final_fairness_report.json")
BOOST_JSON = os.path.join(JSON_DIR, "student_boost_impact.json")
ROUND_LOGS_JSON = os.path.join(JSON_DIR, "sim_rounds.json")


def _ensure_dirs():
    for p in [DATA_DIR, OUTPUT_DIR, JSON_DIR, MODELS_DIR]:
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)


def allocate_all():
    """
    Fast allocation path — uses already-trained models.
    """
    _ensure_dirs()

    students_path = os.path.join(DATA_DIR, "students.csv")
    internships_path = os.path.join(DATA_DIR, "internships.csv")

    if not os.path.exists(students_path):
        raise FileNotFoundError("students.csv missing in /data")
    if not os.path.exists(internships_path):
        raise FileNotFoundError("internships.csv missing in /data")

    students_df = pd.read_csv(students_path)
    internships_df = pd.read_csv(internships_path)

    # Load models + vectorizer
    model_match, model_accept, vectorizer = load_models_and_vectorizer()

    # Build pairs
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
                "pref_1": s.get("pref_1", None),
                "pref_2": s.get("pref_2", None),
                "pref_3": s.get("pref_3", None),
                "pref_4": s.get("pref_4", None),
                "pref_5": s.get("pref_5", None),
                "pref_6": s.get("pref_6", None),
            })

    pairs_df = pd.DataFrame(pairs)

    # Pref rank computation (1-6 else 7)
    def get_pref_rank(row):
        iid = row["internship_id"]
        for r in range(1, 7):
            if row.get(f"pref_{r}") == iid:
                return r
        return 7

    pairs_df["pref_rank"] = pairs_df.apply(get_pref_rank, axis=1)

    # Score pairs
    scored = score_all_pairs(pairs_df, model_match, model_accept, vectorizer)

    # Boost
    boosted = apply_middle_tier_boost(scored)

    # Ranklists
    ranklists = build_ranklists(boosted, internships_df)

    # Allocation
    final_df, round_logs = optionC_allotment_simulated_rejection(
        ranklists=ranklists,
        internships_df=internships_df,
        out_json_dir=JSON_DIR,
        max_rounds=8,
        seed=123,
    )

    # Reports
    fairness_report = build_fairness_report(final_df, students_df, round_logs)

    boost_report = build_student_boost_report(
        boosted_df=boosted,
        final_alloc_df=final_df,
        out_path=BOOST_JSON
    )

    # Save outputs
    final_df.to_csv(FINAL_ALLOC_CSV, index=False)
    with open(FINAL_ALLOC_JSON, "w") as f:
        json.dump(final_df.to_dict(orient="records"), f, indent=2)

    with open(FAIRNESS_JSON, "w") as f:
        json.dump(fairness_report, f, indent=2)

    with open(ROUND_LOGS_JSON, "w") as f:
        json.dump(round_logs, f, indent=2)

    with open(BOOST_JSON, "w") as f:
        json.dump(boost_report, f, indent=2)

    # Combined dashboard
    results = {
        "students": len(students_df),
        "internships": len(internships_df),
        "allocations_count": len(final_df),
        "allocations": final_df.to_dict(orient="records"),
        "fairness": fairness_report,
        "round_logs": round_logs,
        "boost_report": boost_report,
    }

    with open(LAST_RESULTS, "w") as f:
        json.dump(results, f, indent=2)

    return {
        "message": "Allocation completed successfully",
        "summary": {
            "total_students": len(students_df),
            "total_internships": len(internships_df),
            "final_allocations": len(final_df)
        }
    }


def get_dashboard_data():
    if not os.path.exists(LAST_RESULTS):
        raise FileNotFoundError("No results found. Run /admin/allocate first.")
    with open(LAST_RESULTS, "r") as f:
        return json.load(f)


def download_outputs(fname: str):
    """
    Return a file path to be served by FastAPI's FileResponse outside.
    We keep this simple: check both output/ and json_outputs/.
    """
    from fastapi.responses import FileResponse

    # candidate paths
    candidates = [
        os.path.join(OUTPUT_DIR, fname),
        os.path.join(JSON_DIR, fname),
        os.path.join(JSON_DIR, f"{fname}.json"),
        os.path.join(OUTPUT_DIR, f"{fname}.csv"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return FileResponse(p, media_type="application/octet-stream", filename=os.path.basename(p))

    raise FileNotFoundError()
