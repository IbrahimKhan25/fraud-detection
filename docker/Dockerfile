# Build from the repo root:  docker build -f docker/Dockerfile -t fraud-api .
#
# Stages: `base` is the API without a model. The default build (the last
# stage, `api`) is just `base`; the model is mounted at runtime, see
# docker-compose.yml. `release` bakes the model in; the retrain workflow
# builds it with --target release and pushes it to GHCR tagged with the
# model version, so that image runs on its own.
FROM python:3.11-slim AS base

# LightGBM needs OpenMP at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/__init__.py src/__init__.py
COPY src/features src/features
COPY src/monitoring src/monitoring
COPY src/serving src/serving

# The default build mounts the model at runtime (docker-compose.yml); `release` copies it in.
ENV MODEL_PATH=/models/phase2_lgbm.pkl \
    REDIS_URL=redis://redis:6379/0

EXPOSE 8000
# One worker per container: LightGBM already uses all cores per predict call.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS release
COPY models/phase2_lgbm.pkl /models/phase2_lgbm.pkl
# Runs without compose, so it carries compose's thread pin: without it
# LightGBM starts a thread per core in every worker and p99 roughly doubles.
ENV OMP_NUM_THREADS=1

FROM base AS api
