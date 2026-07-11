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
from datetime import datetime

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .config import Config
from .data import collate_fn
from .geo import haversine_km
from .models.net import DeepTTE

MIN_POINTS = 10  # a route-based model needs a route, not just endpoints


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

    app = FastAPI(title="DeepTTE ETA service")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "checkpoint": ckpt_path, "dataset": dataset,
                "hparams": model.hparams}

    @app.post("/predict")
    def predict(req: PredictRequest):
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
