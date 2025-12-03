# ============================
# 1. Base Python Image
# ============================
FROM python:3.10-slim

# Set work directory inside container
WORKDIR /app

# ============================
# 2. Install system dependencies
# ============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================
# 3. Copy project files
# ============================
COPY backend ./backend
COPY src ./src
COPY models ./models
COPY data ./data
COPY output ./output
COPY json_outputs ./json_outputs
COPY requirements.txt ./requirements.txt

# ============================
# 4. Install dependencies
# ============================
RUN pip install --no-cache-dir -r requirements.txt

# ============================
# 5. Expose port
# ============================
EXPOSE 8000

# ============================
# 6. Start Gunicorn + Uvicorn Workers
# ============================
CMD ["gunicorn", "backend.app.main:app", \
     "--workers", "3", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
