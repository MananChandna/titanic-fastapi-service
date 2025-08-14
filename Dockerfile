FROM python:3.9-slim

WORKDIR /app

# Copy the new requirements file
COPY requirements.txt .

# Install all dependencies, including mlflow
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application code
COPY main.py .

# Expose the port the app runs on
EXPOSE 8000

# The command to run the application.
# Note: The MLFLOW_TRACKING_URI must be passed as an environment variable
# to the 'docker run' command.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
