# attrition-ie-mlops
Attrition IE Project for MLOps


mlflow ui --backend-store-uri mlruns --port 5000

python -m pytest -q

uvicorn app:app --reload --port 8000

docker build -t attrition-api:latest .
docker run -p 8000:8000 attrition-api:latest