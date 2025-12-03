"""
optionC_allotment.py

Hybrid multi-round internship seat allocation engine:

✔ Uses ranklists
✔ Uses match_score + accept_score
✔ Simulates acceptance/rejection per offer
✔ Allows students to upgrade if a better-preference internship
  becomes available in later rounds
✔ Produces fairness and simulation logs

Outputs:
 - final_df
 - fairness_report
 - JSON logs:
        sim_rounds.json
        sim_offer_events.json
"""

import os
import json
import random
import pandas as pd
from collections import defaultdict


# ================================================================
# MAIN ALLOTMENT ENGINE
# ================================================================
def optionC_allotment_simulated_rejection(
    ranklists,
    internships_df,
    out_json_dir,
    max_rounds=8,
    default_accept_prob=0.7,
    seed=123,
):
    """
    Realistic seat allocation simulation with acceptance model + upgrades.

    ranklists: { internship_id : [ {student info}, ... ] }
    internships_df: real internships
    """

    random.seed(seed)
    os.makedirs(out_json_dir, exist_ok=True)

    # Seats
    seats = dict(zip(internships_df["internship_id"], internships_df["capacity"]))

    # Track each student's best allocation
    student_alloc = {}            # sid -> internship_id
    student_pref = {}             # sid -> pref_rank

    # Event logs
    offer_events = []
    round_logs = []

    # =====================================================
    # Multi-Round Loop
    # =====================================================
    for rnd in range(1, max_rounds + 1):

        offers_made = 0
        acceptances = 0
        rejections = 0
        upgrades = 0
        filledThisRound = 0

        remaining_at_start = seats.copy()

        # -------------------------------------------------
        # Iterate internship by internship
        # -------------------------------------------------
        for iid, ranked_list in ranklists.items():

            cap = seats.get(iid, 0)
            if cap <= 0:
                continue

            for stu in ranked_list:
                if cap <= 0:
                    break

                sid = stu["student_id"]
                stu_pref = stu["pref_rank"]

                # Skip if student already has a better or equal preference seat
                if sid in student_pref and student_pref[sid] <= stu_pref:
                    continue

                # --------------------------------------------
                # Simulated Accept/Reject
                # --------------------------------------------
                # use accept_score if present else fallback
                p_accept = float(stu.get("accept_score", default_accept_prob))
                accepted = random.random() < p_accept

                offers_made += 1

                if not accepted:
                    rejections += 1
                    offer_events.append({
                        "round": rnd,
                        "student_id": sid,
                        "internship_id": iid,
                        "accepted": False,
                        "reason": "rejected_by_probability"
                    })
                    continue

                # --------------------------------------------
                # If accepted, check if it is an UPGRADE
                # --------------------------------------------
                old_assigned = student_alloc.get(sid)
                old_pref = student_pref.get(sid, 999)

                if old_assigned is not None:
                    # upgrade if new internship is better preference
                    if stu_pref < old_pref:
                        upgrades += 1
                        # release old internship seat
                        seats[old_assigned] += 1
                    else:
                        # no upgrade (worse seat)
                        continue

                # Assign student to new seat
                student_alloc[sid] = iid
                student_pref[sid] = stu_pref

                cap -= 1
                seats[iid] = cap
                acceptances += 1
                filledThisRound += 1

                # Log event
                offer_events.append({
                    "round": rnd,
                    "student_id": sid,
                    "internship_id": iid,
                    "accepted": True,
                    "final_score": stu["final_score"],
                    "pref_rank": stu_pref
                })

        round_logs.append({
            "round": rnd,
            "offers_made": offers_made,
            "acceptances": acceptances,
            "rejections": rejections,
            "upgrades": upgrades,
            "seats_filled_this_round": filledThisRound,
            "seats_available_at_start": remaining_at_start,
        })

        if filledThisRound == 0:
            break

    # =====================================================
    # Final allocations to DataFrame
    # =====================================================
    final_rows = []
    for sid, iid in student_alloc.items():
        final_rows.append({
            "student_id": sid,
            "internship_id": iid,
            "pref_rank": student_pref[sid],
        })

    final_df = pd.DataFrame(final_rows)

    # =====================================================
    # Fairness summary
    # =====================================================
    fairness = _compute_fairness(final_df, ranklists)

    # =====================================================
    # Write JSON logs
    # =====================================================
    with open(os.path.join(out_json_dir, "sim_rounds.json"), "w") as f:
        json.dump(round_logs, f, indent=2)

    with open(os.path.join(out_json_dir, "sim_offer_events.json"), "w") as f:
        json.dump(offer_events, f, indent=2)

    return final_df, fairness


# ================================================================
# FAIRNESS REPORT GENERATION
# ================================================================
def _compute_fairness(final_df, ranklists):

    # Flatten full pool
    full_list = []
    for lst in ranklists.values():
        full_list.extend(lst)
    full_df = pd.DataFrame(full_list)

    # Unique students
    total_applicants = len(full_df["student_id"].unique())
    total_placed = len(final_df["student_id"].unique())

    # Category stats
    cat_stats = {}
    for cat in ["GEN", "OBC", "SC", "ST"]:
        eligible = (full_df["reservation"] == cat).sum()
        selected = full_df.merge(final_df, on="student_id")["reservation"].eq(cat).sum()
        cat_stats[cat] = {
            "eligible": int(eligible),
            "selected": int(selected),
            "rate": 0 if eligible == 0 else selected / eligible,
        }

    # Gender stats
    gender_counts = full_df.merge(final_df, on="student_id")["gender"].value_counts().to_dict()

    # Rural
    rural_eligible = (full_df["rural"] == 1).sum()
    rural_selected = full_df.merge(final_df, on="student_id")["rural"].eq(1).sum()

    return {
        "total_applicants": total_applicants,
        "total_placed": total_placed,
        "placement_rate": total_placed / total_applicants,
        "category_stats": cat_stats,
        "gender_counts_selected": gender_counts,
        "rural": {
            "eligible": int(rural_eligible),
            "selected": int(rural_selected)
        }
    }
