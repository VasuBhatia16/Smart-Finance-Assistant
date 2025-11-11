import os
from dotenv import load_dotenv

load_dotenv()

API_NAME = os.getenv("API_NAME", "Smart Finance Assistant API")

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("API_CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
