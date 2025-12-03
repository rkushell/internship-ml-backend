# fairness_report.py

import pandas as pd

def build_fairness_report(
    final_alloc_df,
    students_df,
    round_logs
):
    """
    final_alloc_df: rows = placed students (unique student-intern)
    students_df: complete student dataset
    round_logs: list of per-round stats from the allocation engine
    """

    # ---------------------------------------
    # BASIC COUNTS
    # ---------------------------------------
    placed_students = final_alloc_df["student_id"].unique().tolist()
    total_placed = len(placed_students)
    total_applicants = len(students_df)

    # Subset of student records of placed students
    placed_df = students_df[students_df["student_id"].isin(placed_students)]

    # ---------------------------------------
    # CATEGORY-WISE FAIRNESS
    # ---------------------------------------
    category_stats = {}
    for cat in ["GEN", "OBC", "SC", "ST"]:
        eligible = len(students_df[students_df["reservation"] == cat])
        placed = len(placed_df[placed_df["reservation"] == cat])

        category_stats[cat] = {
            "eligible": int(eligible),
            "placed": int(placed),
            "placement_rate": round(placed / eligible, 4) if eligible else 0
        }

    # ---------------------------------------
    # GENDER FAIRNESS
    # ---------------------------------------
    gender_counts = placed_df["gender"].value_counts().to_dict()
    gender_counts = {k: int(v) for k, v in gender_counts.items()}

    # ---------------------------------------
    # RURAL FAIRNESS
    # ---------------------------------------
    rural_eligible = len(students_df[students_df["rural"] == 1])
    rural_placed = len(placed_df[placed_df["rural"] == 1])

    rural_stats = {
        "eligible": int(rural_eligible),
        "placed": int(rural_placed),
        "placement_rate": round(rural_placed / rural_eligible, 4)
                           if rural_eligible else 0
    }

    # ---------------------------------------
    # FINAL REPORT DICTIONARY
    # ---------------------------------------
    report = {
        "total_applicants": int(total_applicants),
        "total_placed": int(total_placed),
        "placement_rate": round(total_placed / total_applicants, 4),
        "category_wise": category_stats,
        "gender_wise": gender_counts,
        "rural": rural_stats,
        "round_stats": round_logs
    }

    return report
