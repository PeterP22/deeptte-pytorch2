import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # a fresh (untrained) model is fine for API-contract tests
    from deeptte.models.net import DeepTTE
    ckpt = tmp_path / "model.pt"
    DeepTTE(masked_attention=True).save_checkpoint(ckpt)
    monkeypatch.setenv("DEEPTTE_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("DEEPTTE_DATASET", "chengdu")

    import deeptte.serve as serve
    return TestClient(serve.create_app())


def real_trip_points(n=30):
    trip = json.loads(open("data/test").readline())
    return [[lng, lat] for lng, lat in zip(trip["lngs"][:n], trip["lats"][:n])]


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_eta(client):
    r = client.post("/predict", json={
        "points": real_trip_points(),
        "departure_time": "2026-07-12T18:30:00",
        "driver_id": 3,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["eta_seconds"] == pytest.approx(body["eta_minutes"] * 60, rel=1e-3)
    assert "dist_km" in body


def test_predict_rejects_too_few_points(client):
    r = client.post("/predict", json={
        "points": real_trip_points(3),
        "departure_time": "2026-07-12T18:30:00",
    })
    assert r.status_code == 422


def test_predict_rejects_bad_departure(client):
    r = client.post("/predict", json={
        "points": real_trip_points(),
        "departure_time": "not-a-date",
    })
    assert r.status_code == 422
