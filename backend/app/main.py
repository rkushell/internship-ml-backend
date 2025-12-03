from fastapi import FastAPI
from backend.app.routers.student_api import router as student_router
from backend.app.routers.admin_api import router as admin_router

app = FastAPI(title="Internship ML Backend")

app.include_router(student_router, prefix="/student")
app.include_router(admin_router, prefix="/admin")

@app.get("/")
def root():
    return {"message": "ML Backend API is running"}
