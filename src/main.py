from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib  # Import joblib for loading .pkl files
import numpy as np

app = FastAPI()

# Load your models using joblib
model_1 = joblib.load('data1_model.pkl')
model_2 = joblib.load('data2_model.pkl')
class InputData(BaseModel):
    features: List[float]

@app.post("/predict/")
async def predict(dataset: str, data: InputData):
    # Check if dataset is valid
    if dataset not in ["dataset1", "dataset2"]:
        raise HTTPException(status_code=400, detail="Invalid dataset")

    # Prepare the input data for the model
    features = np.array(data.features).reshape(1, -1)

    # Choose the model based on the dataset
    if dataset == "dataset1":
        model = model_1
    elif dataset == "dataset2":
        model = model_2

    # Perform inference
    prediction = model.predict(features)

    return {"prediction": prediction.tolist()}


