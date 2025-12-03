import pandas as pd
import json
import os

def build_student_boost_report(boosted_df, final_alloc_df, out_path):
    """
    boosted_df must contain:
        student_id, reservation, rural, boost_amount, base_score, boosted_score

    final_alloc_df must contain:
        student_id
    """

    boosted_pairs = boosted_df[boosted_df["boost_amount"] > 0]

    # Student compression
    student_boost = (
        boosted_pairs
        .groupby("student_id")
        .agg(
            max_boost_amt=("boost_amount", "max"),
            reservation=("reservation", "first"),
            rural=("rural", "first"),
            pre_boost_best=("base_score", "max"),
            post_boost_best=("boosted_score", "max")
        )
        .reset_index()
    )

    total_boosted_students = len(student_boost)

    placed_students = set(final_alloc_df["student_id"].unique())

    boosted_selected_students = student_boost[
        student_boost["student_id"].isin(placed_students)
    ]
    boosted_not_selected_students = student_boost[
        ~student_boost["student_id"].isin(placed_students)
    ]

    uplift_success_count = len(boosted_selected_students)

    # -------------------------------------------------------
    # Category level uplift success
    # -------------------------------------------------------
    category_success = (
        boosted_selected_students["reservation"]
        .value_counts().to_dict()
    )
    category_success = {k: int(v) for k, v in category_success.items()}

    rural_uplift_success = int(
        boosted_selected_students[boosted_selected_students["rural"] == 1].shape[0]
    )

    # -------------------------------------------------------
    # Counterfactual uplift attribution logic
    # -------------------------------------------------------
    # If post_boost_best > pre_boost_best significantly, assume boost helped placement
    # Threshold = 0.01 (customisable)
    THRESH = 0.01

    boosted_selected_students["counterfactual_helped"] = (
        boosted_selected_students["post_boost_best"] -
        boosted_selected_students["pre_boost_best"]
    ) > THRESH

    helped_count = int(boosted_selected_students["counterfactual_helped"].sum())

    category_attribution = (
        boosted_selected_students[boosted_selected_students["counterfactual_helped"] == True]
        ["reservation"].value_counts().to_dict()
    )
    category_attribution = {k: int(v) for k, v in category_attribution.items()}

    rural_attribution = int(
        boosted_selected_students[
            (boosted_selected_students["counterfactual_helped"] == True) &
            (boosted_selected_students["rural"] == 1)
        ].shape[0]
    )

    avg_student_boost = float(student_boost["max_boost_amt"].mean())
    max_student_boost = float(student_boost["max_boost_amt"].max())
    coverage_ratio = total_boosted_students / boosted_df["student_id"].nunique()

    report = {
        "boosted_students": int(total_boosted_students),

        "boosted_selected": uplift_success_count,
        "boosted_not_selected": int(total_boosted_students - uplift_success_count),
        "uplift_success_rate": round(uplift_success_count / total_boosted_students, 4),

        "category_uplift_success": category_success,
        "rural_uplift_success": rural_uplift_success,

        "avg_student_boost": round(avg_student_boost, 4),
        "max_boost_per_student": round(max_student_boost, 4),
        "coverage_ratio": round(coverage_ratio, 4),

        # --------------------------------
        # Counterfactual uplift attribution
        # --------------------------------
        "counterfactual_helped_students": helped_count,
        "counterfactual_help_rate_among_boosted": round(helped_count / total_boosted_students, 4),

        "category_counterfactual_help": category_attribution,
        "rural_counterfactual_help": rural_attribution
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
