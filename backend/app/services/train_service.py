import pandas as pd
from src.models import train_models
from src.data_real_past_generator import generate_pseudo_past_data

DATA_DIR = "data"

def train_all():
    students = pd.read_csv(f"{DATA_DIR}/students.csv")
    internships = pd.read_csv(f"{DATA_DIR}/internships.csv")

    # Generate synthetic past data
    past_df = generate_pseudo_past_data(
        students_df=students,
        internships_df=internships,
        n_samples=10000,
        seed=42,
        save_path=f"{DATA_DIR}/past_pairs_gen.csv"
    )

    # Train models
    train_models(
        past_df=past_df,
        students_df=students,
        internships_df=internships,
        seed=42
    )

    return {"message": "Training completed successfully"}
