FROM jupyter/scipy-notebook:python-3.9

# Switch to the root user to install new packages
USER root

WORKDIR /app

# Copy the simplified requirements file
COPY requirements.txt .

# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training data and the training script
COPY train.csv .
COPY train_model.py .

# --- THIS IS THE CRITICAL NEW STEP ---
# Run the training script to create a compatible model.joblib file
RUN python3 train_model.py

# Copy the main application code
COPY main.py .

# Expose the port the app runs on
EXPOSE 8000

# The command to run the application.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
