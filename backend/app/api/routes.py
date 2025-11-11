from fastapi import APIRouter
from app.schemas.predict import PredictRequest, PredictResponse, ForecastPoint
from app.utils.date_utils import add_months

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    history = req.history

    # requires history of minimum 3 months to predict
    if len(history) < 3:
        return PredictResponse(
            forecast=[], note="At least 3 months of data required."
        )

    # last year-month
    last_year, last_month = map(int, history[-1].month.split("-"))

    # average of last 3 months total expenses
    last_three = history[-3:]
    totals = [sum(h.categories.values()) for h in last_three]
    avg_expense = sum(totals) / 3

    forecast_list = []

    for i in range(1, req.horizon + 1):
        y, m = add_months(last_year, last_month, i)
        f_month = f"{y:04d}-{m:02d}"

        projected_expense = avg_expense
        projected_savings = history[-1].income - projected_expense

        forecast_list.append(
            ForecastPoint(
                month=f_month,
                total_expense=round(projected_expense, 2),
                projected_savings=round(projected_savings, 2)
            )
        )

    return PredictResponse(
        forecast=forecast_list,
        note="Baseline moving-average forecast. ML model integration pending."
    )
