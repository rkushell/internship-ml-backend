import os

DATA_DIR = "data"

def upload_students_csv(file):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, "students.csv")
    with open(file_path, "wb") as f:
        f.write(file.file.read())

def upload_internships_csv(file):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, "internships.csv")
    with open(file_path, "wb") as f:
        f.write(file.file.read())
