from fastapi import FastAPI
from mangum import Mangum
from .api.models.request import PredictRequest, PredictResponse
from .model.predictor import PredictionModel


app = FastAPI(title="Sentiment Inference API")
handler = Mangum(app)

predictor = PredictionModel()


@app.get("/")
def welcome_root():
    return {"message": "Welcome from sentiment analysis API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    label = predictor.predict(request.text)
    return PredictResponse(prediction=label)