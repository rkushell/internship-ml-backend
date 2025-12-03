import os

DATA_DIR = "data"


def _ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def upload_students_csv(file):
    """
    Save UploadFile to data/students.csv
    """
    _ensure_data_dir()
    file_path = os.path.join(DATA_DIR, "students.csv")
    # file.file is SpooledTemporaryFile (binary). Read bytes and write.
    with open(file_path, "wb") as f:
        f.write(file.file.read())


def upload_internships_csv(file):
    _ensure_data_dir()
    file_path = os.path.join(DATA_DIR, "internships.csv")
    with open(file_path, "wb") as f:
        f.write(file.file.read())
