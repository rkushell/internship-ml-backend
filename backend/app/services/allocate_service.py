import pandas as pd
from src.models import load_models_and_vectorizer, score_all_pairs
from src.ranklist_builder import build_ranklists
from src.optionC_allotment import optionC_allotment_simulated_rejection
from src.fairness_report import build_fairness_report

DATA_DIR = "data"
OUTPUT_DIR = "output"
JSON_DIR = "json_outputs"

def allocate_all():
    students = pd.read_csv(f"{DATA_DIR}/students.csv")
    internships = pd.read_csv(f"{DATA_DIR}/internships.csv")

    # Step 1 — Build all student-internship pairs
    pairs = []
    for _, s in students.iterrows():
        for _, j in internships.iterrows():
            pairs.append({
                "student_id": s["student_id"],
                "internship_id": j["internship_id"],
                "skills": s["skills"],
                "req_skills_job": j["req_skills"],
                "gpa": s["gpa"],
                "stipend_internship": j["stipend"],
                "reservation": s["reservation"],
                "gender": s["gender"],
                "rural": s["rural"],
                "pref_rank": 1,
            })
    pairs_df = pd.DataFrame(pairs)

    # Step 2 — Score pairs using existing model
    model_match, model_accept, vectorizer = load_models_and_vectorizer()
    scored_pairs = score_all_pairs(pairs_df, model_match, model_accept, vectorizer)

    # Step 3 — Build ranklists
    ranklists = build_ranklists(scored_pairs, internships)

    # Step 4 — Run allocator
    final_df, round_logs = optionC_allotment_simulated_rejection(
        ranklists, internships, JSON_DIR
    )

    final_df.to_csv(f"{OUTPUT_DIR}/final_allocations.csv", index=False)

    fairness = build_fairness_report(final_df, students, round_logs)

    return {
        "message": "Allocation completed",
        "allocated": len(final_df),
        "fairness": fairness
    }


def get_dashboard_data():
    final = pd.read_csv(f"{OUTPUT_DIR}/final_allocations.csv")
    return final.to_dict(orient="records")
