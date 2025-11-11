from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import API_NAME, CORS_ORIGINS
from app.api.routes import router

app = FastAPI(title=API_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "Smart Finance Assistant API running"}
