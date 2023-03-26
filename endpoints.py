import os
from spam_classifier.Models.NaiveBayes import NaiveBayes
from fastapi import FastAPI
from pydantic import BaseModel

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


@app.get("/")
async def read_root():
    return {"message": "Hello World!"}


@app.post("/predict", response_model=Input)
async def predict_text(text: Input):
    output = model.predict_text(text)
    return output


@app.get("/fetch_scores", response_model=Output)
async def fetch_scores():

    return model.get_score()
