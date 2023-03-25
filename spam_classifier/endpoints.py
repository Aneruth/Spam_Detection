import os
import pickle

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


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


# TODO: fix the path for the model fetching and loading the model error handling
def read_model():
    # check if the model is present in the directory
    # check for the pickle file in the directory
    files = os.listdir('../spam_classifier/trained_models')
    name = None
    for file in files:
        if file.startswith('model'):
            name = os.path.join('../spam_classifier/trained_models', file)
            break

    # check if the path exists
    if os.path.exists(name):
        # read the model from the pickle file
        print("Model path exists")
        model = pickle.load(open(name, "rb"))
        return model
    return Exception("Model not found")


@app.post("/predict")
async def predict_text(text: Input):
    # read the model from the pickle file
    model = read_model()
    if isinstance(model, Exception):
        return "Sorry run the model first"

    # get the prediction
    text_transform = model.transform_text(text)
    prediction = model.predict(text_transform)

    # get the score
    score = model.get_individual_score(text)
    return score
