FROM python:3.11-slim

WORKDIR /app

# Install OS deps if needed (for xgboost / numpy)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model
COPY app.py .
COPY xgb_attrition_pipeline.joblib .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
