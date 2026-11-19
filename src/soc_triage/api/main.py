import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from soc_triage.api.predictor import Predictor
from soc_triage.api.schemas import PredictRequest, PredictResponse
from soc_triage.utils.config import PROJECT_ROOT, load_config
from soc_triage.utils.exceptions import DataValidationError
from soc_triage.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = PROJECT_ROOT / os.getenv("MODEL_PATH", "artifacts/model.joblib")
    app.state.predictor = Predictor(model_path, load_config())
    logger.info("Model loaded from %s", model_path)
    yield


app = FastAPI(title="SOC incident triage", version="0.1.0", lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "SOC incident triage API. Open /docs to try POST /predict."}


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "predictor")}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    rows = [item.model_dump() for item in body.evidence]
    try:
        results = request.app.state.predictor.predict_incidents(rows)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    return {"results": results}
