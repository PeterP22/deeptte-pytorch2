# DeepTTE ETA service — CPU-only torch (the CUDA wheel is ~5GB of dead weight here)
FROM python:3.12-slim

WORKDIR /app

# CPU torch first (its own index), then the package deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy tqdm "fastapi>=0.111" "uvicorn>=0.30"

COPY deeptte/ deeptte/
# committed deploy artifacts: trained weights + the dataset's normalization
# stats (serving never needs the trips themselves)
COPY deploy/stats.json data/porto/stats.json
COPY deploy/model.pt model/best.pt

ENV DEEPTTE_CHECKPOINT=/app/model/best.pt \
    DEEPTTE_DATASET=porto

EXPOSE 8000
CMD uvicorn deeptte.serve:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}
