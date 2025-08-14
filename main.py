from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import os

# --- Step 1: Initialize the FastAPI app ---
app = FastAPI(title="Titanic Survival Prediction API",
              description="An API to predict passenger survival on the Titanic, with models served from MLflow.",
              version="1.0")

# --- Step 2: Load the Production model from MLflow Model Registry ---
# This is loaded once when the application starts.
# It requires the MLFLOW_TRACKING_URI environment variable to be set.
model = None
model_name = "production-titanic-predictor" # The name of the model in the MLflow Registry
model_stage = "Production"

@app.on_event("startup")
def load_model():
    global model
    try:
        # Check if MLFLOW_TRACKING_URI is set
        if 'MLFLOW_TRACKING_URI' not in os.environ:
            raise EnvironmentError("MLFLOW_TRACKING_URI environment variable not set.")
        
        print(f"Loading model '{model_name}' version '{model_stage}' from registry...")
        
        # Load the model from the registry
        model_uri = f"models:/{model_name}/{model_stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # The app will start, but the /predict endpoint will return an error.
        model = None

# --- Step 3: Define the input data model (Pydantic) ---
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float

    class Config:
        schema_extra = {
            "example": {
                "Pclass": 3, "Sex": "male", "Age": 22.0,
                "SibSp": 1, "Parch": 0, "Fare": 7.25
            }
        }

# --- Step 4: Define the prediction endpoint ---
@app.post("/predict", tags=["Prediction"])
async def predict_survival(passenger: Passenger):
    """
    Predicts whether a passenger would have survived the Titanic disaster.
    """
    if model is None:
        return {"error": "Model not loaded. Please check server logs."}

    # Convert the Pydantic model to a pandas DataFrame
    data = pd.DataFrame([passenger.dict()])

    # Make a prediction
    prediction_result = model.predict(data)
    
    # The output from a scikit-learn pipeline is often a numpy array
    survival_status = "Survived" if prediction_result[0] == 1 else "Did not survive"
    
    return {
        "prediction": survival_status
    }

# --- Step 5: Root endpoint for health checks ---
@app.get("/", tags=["Health Check"])
async def root():
    if model is not None:
        return {"message": "API is running and model is loaded."}
    else:
        return {"message": "API is running, but model failed to load."}
