import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("AI_BACKEND", "stub")
os.environ["MDR_DB"] = os.path.join(os.path.dirname(__file__), "test_mdr.sqlite3")

import pytest
from fastapi.testclient import TestClient

from app_v2 import app

SCORER = {"X-API-Key": "demo-scorer"}
VIEWER = {"X-API-Key": "demo-viewer"}
ADMIN = {"X-API-Key": "demo-admin"}


@pytest.fixture(scope="session")
def client():
    if os.path.exists(os.environ["MDR_DB"]):
        os.remove(os.environ["MDR_DB"])
    return TestClient(app)
