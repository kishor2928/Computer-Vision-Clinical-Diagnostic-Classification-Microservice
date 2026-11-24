import io
import numpy as np
import cv2
import pytest
from app.main import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_dummy_jpeg():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return io.BytesIO(buf.tobytes())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert "status" in data
    assert "cnn_loaded" in data


def test_predict_missing_image(client):
    r = client.post("/predict")
    assert r.status_code == 400
    assert r.get_json()["error"] == "missing_image"


def test_predict_cnn_smoke(client):
    data = {
        "model": "cnn",
    }
    files = {
        "image": ( _make_dummy_jpeg(), "test.jpg" ),
    }
    # flask test client wants multipart a bit differently
    r = client.post(
        "/predict",
        data={"model": "cnn", "image": ( _make_dummy_jpeg(), "test.jpg" )},
        content_type="multipart/form-data",
    )
    # model may not have real weights, but the endpoint should still respond
    assert r.status_code in (200, 503)
    body = r.get_json()
    assert "request_id" in body or "error" in body
