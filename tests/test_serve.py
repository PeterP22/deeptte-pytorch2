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


def test_demo_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "DEEP" in r.text
    assert client.get("/presets.js").status_code == 200


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


def test_predict_rejects_out_of_coverage(client):
    sydney = [[151.25 + i * 0.001, -33.89 + i * 0.001] for i in range(12)]
    r = client.post("/predict", json={
        "points": sydney,
        "departure_time": "2026-07-12T18:15:00",
    })
    assert r.status_code == 422
    assert "coverage" in r.json()["detail"]


def test_rate_limit(tmp_path, monkeypatch):
    from deeptte.models.net import DeepTTE
    ckpt = tmp_path / "model.pt"
    DeepTTE().save_checkpoint(ckpt)
    monkeypatch.setenv("DEEPTTE_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("DEEPTTE_RATE_LIMIT", "3")
    monkeypatch.setenv("DEEPTTE_DATASET", "chengdu")
    import importlib
    import deeptte.serve as serve
    importlib.reload(serve)  # RATE_LIMIT is read at import time
    limited = TestClient(serve.create_app())
    payload = {"points": real_trip_points(), "departure_time": "2026-07-12T18:15:00"}
    codes = [limited.post("/predict", json=payload).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429 and codes[4] == 429
    importlib.reload(serve)  # restore module-level default for other tests
