"""
data_generator.py

Generates synthetic data for:
- students.csv
- internships.csv
- past_pairs.csv

The synthetic data simulates:
- student skills, GPA, reservation, gender, rural background
- internship required skills, capacity, stipend, sector, location_type
- past match + accept records for model training
"""

import os
import random
import numpy as np
import pandas as pd

# -----------------------------
# Global config lists
# -----------------------------
SKILLS = [
    "python", "java", "sql", "excel", "cloud", "networking", "frontend",
    "backend", "design", "presentation", "writing", "autocad",
    "manufacturing", "analysis", "ml"
]

SECTORS = [
    "Finance", "Marketing", "Electronics", "Mechanical",
    "IT Services", "Healthcare", "Automobile"
]

LOCATION_TYPES = ["Office", "Factory", "Remote", "Hybrid"]

RESERVATIONS = ["GEN", "OBC", "SC", "ST"]
GENDERS = ["M", "F", "O"]


# ----------------------------------------------------------------
# STUDENT GENERATION
# ----------------------------------------------------------------
def generate_students(n_students=2000, n_internships=40):
    """
    Generate synthetic student dataset.
    Returns a DataFrame.
    """
    df = pd.DataFrame()
    df["student_id"] = [f"S{10000+i}" for i in range(n_students)]

    # Random skills per student
    df["skills"] = df.apply(
        lambda _: " ".join(random.sample(SKILLS, random.randint(3, 6))),
        axis=1
    )

    # Academic + demographic features
    df["gpa"] = np.round(np.random.uniform(5.0, 9.8, n_students), 2)
    df["gender"] = np.random.choice(GENDERS, n_students, p=[0.45, 0.45, 0.10])
    df["reservation"] = np.random.choice(
        RESERVATIONS, n_students,
        p=[0.55, 0.28, 0.10, 0.07]
    )
    df["rural"] = np.random.choice([0, 1], n_students, p=[0.75, 0.25])

    # Random preferences from internship list
    internship_ids = [f"I{str(i+1).zfill(3)}" for i in range(n_internships)]

    for p in range(1, 7):
        df[f"pref_{p}"] = df.apply(
            lambda _: random.choice(internship_ids),
            axis=1
        )

    return df


# ----------------------------------------------------------------
# INTERNSHIP GENERATION
# ----------------------------------------------------------------
def generate_internships(n=40):
    """
    Create internship dataset:
    - Required skills
    - Stipend
    - Sector
    - Location type
    - Capacity
    """
    df = pd.DataFrame()
    df["internship_id"] = [f"I{str(i+1).zfill(3)}" for i in range(n)]

    df["sector"] = [random.choice(SECTORS) for _ in range(n)]
    df["tier"] = [random.choice(["Tier1", "Tier2", "Tier3"]) for _ in range(n)]
    df["location_type"] = [random.choice(LOCATION_TYPES) for _ in range(n)]

    # Stipend = 4500 base + (0, 500, 1000)
    df["stipend"] = [4500 + random.choice([0, 500, 1000]) for _ in range(n)]

    # Seats per internship
    df["capacity"] = np.random.randint(10, 25, n)

    # Required skills
    df["req_skills"] = df.apply(
        lambda _: " ".join(random.sample(SKILLS, random.randint(3, 5))),
        axis=1
    )

    return df


# ----------------------------------------------------------------
# PAST MATCH / ACCEPT DATA GENERATION
# ----------------------------------------------------------------
def generate_past_data(students_df, internships_df, n_samples=3000):
    """
    Generate training data for ML models.
    Each row represents (student, internship) with:
        match ∈ {0,1}
        accept ∈ {0,1}
    """
    student_ids = students_df["student_id"].tolist()
    internship_ids = internships_df["internship_id"].tolist()

    rows = []

    for _ in range(n_samples):
        sid = random.choice(student_ids)
        jid = random.choice(internship_ids)

        srow = students_df[students_df["student_id"] == sid].iloc[0]
        jrow = internships_df[internships_df["internship_id"] == jid].iloc[0]

        student_skillset = set(srow["skills"].split())
        job_skillset = set(jrow["req_skills"].split())
        overlap = len(student_skillset.intersection(job_skillset))

        # Probability model
        base_prob = max(
            0.01,
            overlap * 0.15 + (srow["gpa"] - 5) * 0.05
        )

        match = int(random.random() < base_prob)
        accept = int(match and (random.random() < 0.7))

        rows.append({
            "student_id": sid,
            "internship_id": jid,
            "skills": srow["skills"],
            "req_skills_job": jrow["req_skills"],
            "gpa": srow["gpa"],
            "stipend_internship": jrow["stipend"],
            "reservation": srow["reservation"],
            "gender": srow["gender"],
            "rural": srow["rural"],
            "match": match,
            "accept": accept
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------
# WRAPPER — Generate all & write files
# ----------------------------------------------------------------
def generate_synthetic_data(out_dir="data", n_students=2000, n_internships=40):
    """
    Generates:
        students.csv
        internships.csv
        past_pairs.csv

    Returns:
        students_df, internships_df, past_df
    """
    os.makedirs(out_dir, exist_ok=True)

    students_df = generate_students(n_students, n_internships)
    internships_df = generate_internships(n_internships)
    past_df = generate_past_data(students_df, internships_df, n_samples=3000)

    students_df.to_csv(os.path.join(out_dir, "students.csv"), index=False)
    internships_df.to_csv(os.path.join(out_dir, "internships.csv"), index=False)
    past_df.to_csv(os.path.join(out_dir, "past_pairs.csv"), index=False)

    return students_df, internships_df, past_df
