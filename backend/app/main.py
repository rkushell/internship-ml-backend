from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers.student_api import router as student_router
from backend.app.routers.admin_api import router as admin_router
from backend.app.routers.upload_api import router as upload_router
from backend.app.routers.dashboard_api import router as dashboard_router

app = FastAPI(title="PM Internship Prediction API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ROUTERS
app.include_router(student_router,   prefix="/predict",         tags=["Prediction"])
app.include_router(admin_router,     prefix="/admin",           tags=["Admin"])
app.include_router(upload_router,    prefix="/admin/upload",    tags=["Admin Upload"])
app.include_router(dashboard_router, prefix="/admin/dashboard", tags=["Admin Dashboard"])


@app.get("/")
def root():
    return {"message": "API is running"}
