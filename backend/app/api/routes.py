from fastapi import APIRouter, HTTPException
from app.schemas.predict import PredictRequest, PredictResponse, ForecastPoint
from app.utils.date_utils import add_months


from ml.predict_new import Predictor


router = APIRouter()



@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict future monthly expenses and projected savings using trained LSTM.
    Includes category-level dynamic breakdown.
    """

    history = req.history
    model_type = req.model_type or "lstm"
    predictor = Predictor(model_type=model_type)
    if len(history) < 6:
        return PredictResponse(
            forecast=[],
            note="At least 6 months of historical data required for ML prediction."
        )

    hist_dicts = [h.dict() for h in history]

    
    try:
        ml_result = predictor.predict_next_month(hist_dicts, window=6)
        predicted_total = float(ml_result["predicted_total_expenses"])
        breakdown = ml_result["category_breakdown"]  # dict[str, float]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

    forecast_list = []

    last_year, last_month = map(int, history[-1].month.split("-"))

    for i in range(1, req.horizon + 1):

        
        f_year, f_month = add_months(last_year, last_month, i)
        f_month_str = f"{f_year:04d}-{f_month:02d}"

        
        total_expense = predicted_total
        projected_savings = history[-1].income - total_expense

        
        forecast_list.append(
            ForecastPoint(
                month=f_month_str,
                total_expense=round(total_expense, 2),
                projected_savings=round(projected_savings, 2),
                category_breakdown={
                    cat: round(val, 2) for cat, val in breakdown.items()
                }
            )
        )
    if req.model_type == "lstm":
        note = "Forecast generated using trained LSTM model with dynamic category breakdown."
    else:
        note = "Forecast generated using trained XGBoost model with dynamic category breakdown."
    return PredictResponse(
        forecast=forecast_list,
        note=note
    )
