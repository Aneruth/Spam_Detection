import os
from spam_classifier.Models.NaiveBayes import NaiveBayes
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi

app = FastAPI()
model = NaiveBayes()
model.run_model()


class Input(BaseModel):
    text: str


class Output(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float


class predictOutput(BaseModel):
    prediction: str


class ConfusionMatrix(BaseModel):
    TP: int
    FP: int
    FN: int
    TN: int


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Custom API Index",
        version="1.0.0",
        description="This is a custom API index page for my FastAPI app.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/")
async def read_root():
    return {"message": "Hello World!"}


@app.post("/predict", response_model=predictOutput)
async def predict_text(input_data: Input):
    text = input_data.text
    if text is None:
        return {"message": "Please enter a text"}
    elif isinstance(text, str) is False:
        return {"message": "Please enter a valid text"}
    output = model.predict_text(text)
    return {"prediction": output}


@app.get("/fetch_scores", response_model=Output)
async def fetch_scores():
    return model.get_score()


@app.get("/fetch_confusion_matrix", response_model=ConfusionMatrix)
async def fetch_confusion_matrix():
    return model.get_confusion_matrix()
