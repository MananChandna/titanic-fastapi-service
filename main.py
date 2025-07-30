# File: main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize the FastAPI app
app = FastAPI(title="Titanic Survival Prediction API",
              description="An API to predict passenger survival on the Titanic.",
              version="1.0")

# 2. Load the trained model
# This is loaded once when the application starts
try:
    model = joblib.load('model.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.joblib not found. Please run train_model.py first.")
    model = None

# 3. Define the input data model using Pydantic
# This ensures that any incoming request has the correct data types and fields.
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float

    # Example data for the interactive docs
    class Config:
        schema_extra = {
            "example": {
                "Pclass": 3,
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25
            }
        }

# 4. Define the prediction endpoint
@app.post("/predict", tags=["Prediction"])
async def predict_survival(passenger: Passenger):
    """
    Predicts whether a passenger would have survived the Titanic disaster.
    - **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
    - **Sex**: Sex (male or female)
    - **Age**: Age in years
    - **SibSp**: # of siblings / spouses aboard the Titanic
    - **Parch**: # of parents / children aboard the Titanic
    - **Fare**: Passenger fare
    """
    if model is None:
        return {"error": "Model not loaded. Please check server logs."}

    # Convert the Pydantic model to a pandas DataFrame
    data = pd.DataFrame([passenger.dict()])

    # Make a prediction
    prediction = model.predict(data)
    probability = model.predict_proba(data)

    # Interpret the prediction
    survival_status = "Survived" if prediction[0] == 1 else "Did not survive"
    survival_probability = probability[0][1] # Probability of 'Survived' class
    
    return {
        "prediction": survival_status,
        "probability_of_survival": f"{survival_probability:.2%}"
    }

# Optional: A root endpoint for health checks
@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "API is running."}