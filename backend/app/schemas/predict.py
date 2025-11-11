from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class MonthlyRecord(BaseModel):
    month: str = Field(..., pattern=r"^\d{4}-\d{2}$", description="YYYY-MM")
    income: float = Field(..., ge=0)
    savings_goal: float = Field(..., ge=0)
    categories: Dict[str, float] = Field(...)


class PredictRequest(BaseModel):
    history: List[MonthlyRecord] = Field(..., min_items=3)
    horizon: int = Field(3, ge=1, le=12)


class ForecastPoint(BaseModel):
    month: str
    total_expense: float
    projected_savings: float
    category_breakdown: Optional[Dict[str, float]] = None


class PredictResponse(BaseModel):
    forecast: List[ForecastPoint]
    note: str | None = None
