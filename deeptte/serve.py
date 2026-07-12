"""FastAPI serving layer for a trained DeepTTE checkpoint.

The client sends the planned route (GPS points) and a departure time; the
server derives everything else the model needs (cumulative distances via
haversine, day-of-week / time-of-day IDs) — mirroring how a production ETA
service enriches a request before hitting the model.

Env vars:
    DEEPTTE_CHECKPOINT  path to a checkpoint (default checkpoints/porto-t1/best.pt)
    DEEPTTE_DATASET     which Config stats to use (default porto)

Run locally:  uv run uvicorn "deeptte.serve:create_app" --factory --port 8000
"""
import os
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from .config import Config
from .data import collate_fn
from .geo import haversine_km
from .models.net import DeepTTE

MIN_POINTS = 8  # matches the training data's minimum (porto_prepare MIN_POINTS)
COVERAGE_SIGMA = 30  # points beyond mean ± 30σ of the training coords are out of coverage
RATE_LIMIT = int(os.environ.get("DEEPTTE_RATE_LIMIT", "30"))  # predictions / minute / IP


class PredictRequest(BaseModel):
    points: list[list[float]] = Field(
        ..., description="Planned route as [[lng, lat], ...], ordered origin->destination"
    )
    departure_time: datetime
    driver_id: int = 0

    @field_validator("points")
    @classmethod
    def check_points(cls, pts):
        if len(pts) < MIN_POINTS:
            raise ValueError(f"need at least {MIN_POINTS} route points, got {len(pts)}")
        if any(len(p) != 2 for p in pts):
            raise ValueError("each point must be [lng, lat]")
        return pts


def build_trip(req: PredictRequest):
    """Assemble the trip dict the data pipeline expects. `time` and `time_gap`
    are labels/training-only signals — dummied out at inference."""
    lngs = [p[0] for p in req.points]
    lats = [p[1] for p in req.points]
    dist_gap = [0.0]
    for (ln1, la1), (ln2, la2) in zip(req.points, req.points[1:]):
        dist_gap.append(dist_gap[-1] + haversine_km(ln1, la1, ln2, la2))
    dep = req.departure_time
    return {
        "driverID": req.driver_id % 24000,  # clamp to embedding vocab
        "dateID": dep.day - 1,
        "weekID": dep.weekday(),
        "timeID": dep.hour * 60 + dep.minute,
        "dist": dist_gap[-1],
        "time": 0.0,
        "lngs": lngs,
        "lats": lats,
        "states": [1.0] * len(lngs),
        "time_gap": [0.0] * len(lngs),
        "dist_gap": dist_gap,
    }


def create_app() -> FastAPI:
    ckpt_path = os.environ.get("DEEPTTE_CHECKPOINT", "checkpoints/porto-t1/best.pt")
    dataset = os.environ.get("DEEPTTE_DATASET", "porto")

    config = Config.for_dataset(dataset)
    model = DeepTTE.from_checkpoint(ckpt_path)
    model.eval()

    # a model only knows the city it was trained on — refuse the rest. Prefer
    # the training data's actual bounding box (from stats.json), padded a
    # little; fall back to σ-based bounds for datasets without one.
    if config.coverage:
        pad = 0.05  # degrees, ~5 km of slack around the training box
        lng_lo, lng_hi = config.coverage["lng"][0] - pad, config.coverage["lng"][1] + pad
        lat_lo, lat_hi = config.coverage["lat"][0] - pad, config.coverage["lat"][1] + pad
    else:
        lng_lo = config.mean("lngs") - COVERAGE_SIGMA * config.std("lngs")
        lng_hi = config.mean("lngs") + COVERAGE_SIGMA * config.std("lngs")
        lat_lo = config.mean("lats") - COVERAGE_SIGMA * config.std("lats")
        lat_hi = config.mean("lats") + COVERAGE_SIGMA * config.std("lats")

    hits: dict[str, deque] = defaultdict(deque)

    def check_rate(request: Request):
        ip = (request.headers.get("x-forwarded-for") or
              (request.client.host if request.client else "unknown")).split(",")[0].strip()
        now = time.monotonic()
        q = hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            raise HTTPException(status_code=429,
                                detail=f"rate limit: {RATE_LIMIT} predictions/minute per client")
        q.append(now)
        if len(hits) > 10_000:  # bound memory under IP churn
            hits.clear()

    app = FastAPI(title="DeepTTE ETA service")
    static = Path(__file__).parent / "static"

    @app.get("/", include_in_schema=False)
    def demo():
        return FileResponse(static / "index.html")

    @app.get("/presets.js", include_in_schema=False)
    def presets():
        return FileResponse(static / "presets.js")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "checkpoint": ckpt_path, "dataset": dataset,
                "hparams": model.hparams,
                "coverage": {"lng": [round(lng_lo, 3), round(lng_hi, 3)],
                             "lat": [round(lat_lo, 3), round(lat_hi, 3)]}}

    @app.post("/predict")
    def predict(req: PredictRequest, request: Request):
        check_rate(request)
        for lng, lat in req.points:
            if not (lng_lo <= lng <= lng_hi and lat_lo <= lat <= lat_hi):
                raise HTTPException(
                    status_code=422,
                    detail=f"point ({lng}, {lat}) is outside model coverage — this model "
                           f"was trained on {dataset} (lng {lng_lo:.2f}..{lng_hi:.2f}, "
                           f"lat {lat_lo:.2f}..{lat_hi:.2f})")
        trip = build_trip(req)
        attr, traj = collate_fn([trip], config)
        with torch.no_grad():
            pred = model(attr, traj, config)  # eval mode: normalized total time
        eta_s = float(config.unnormalize(pred, "time").squeeze())
        if not (0 < eta_s < 86400):
            raise HTTPException(status_code=500, detail=f"implausible eta {eta_s:.0f}s")
        return {
            "eta_seconds": round(eta_s, 1),
            "eta_minutes": round(eta_s / 60, 2),
            "dist_km": round(trip["dist"], 3),
        }

    return app
