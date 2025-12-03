import os
import pandas as pd

from src.data_real_past_generator import generate_pseudo_past_data
from src.models import train_models

DATA_DIR = "data"
MODELS_DIR = "models"


def train_all(n_samples_past: int = 12000, generator_seed: int = 123, train_seed: int = 42):
    """
    Trains match + accept models using students.csv + internships.csv.
    This is CPU/memory intensive — run locally if possible.
    """
    if not os.path.exists(os.path.join(DATA_DIR, "students.csv")):
        raise FileNotFoundError("students.csv missing in /data")

    if not os.path.exists(os.path.join(DATA_DIR, "internships.csv")):
        raise FileNotFoundError("internships.csv missing in /data")

    students_df = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    internships_df = pd.read_csv(os.path.join(DATA_DIR, "internships.csv"))

    # Generate pseudo past pairs (saved to data/past_pairs_gen.csv)
    past_path = os.path.join(DATA_DIR, "past_pairs_gen.csv")
    past_df = generate_pseudo_past_data(
        students_df=students_df,
        internships_df=internships_df,
        n_samples=n_samples_past,
        seed=generator_seed,
        save_path=past_path
    )

    # Train models (saves to models/ folder)
    train_models(
        past_df=past_df,
        students_df=students_df,
        internships_df=internships_df,
        seed=train_seed
    )

    return {
        "message": "Training completed successfully",
        "students": len(students_df),
        "internships": len(internships_df),
        "past_pairs_generated": len(past_df),
        "models_dir": MODELS_DIR,
    }
