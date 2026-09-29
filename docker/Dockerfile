# Build from the repo root:  docker build -f docker/Dockerfile -t fraud-api .
FROM python:3.11-slim

# LightGBM needs OpenMP at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/__init__.py src/__init__.py
COPY src/features src/features
COPY src/serving src/serving

# The model is mounted at runtime (see docker-compose.yml), not baked in.
ENV MODEL_PATH=/models/phase2_lgbm.pkl \
    REDIS_URL=redis://redis:6379/0

EXPOSE 8000
# One worker per container: LightGBM already uses all cores per predict call.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
