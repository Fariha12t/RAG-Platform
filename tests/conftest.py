import os
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="rag_tests_")
os.environ["DATA_DIR"] = _tmp
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["LLM_PROVIDER"] = "extractive"
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["ADMIN_API_KEY"] = "test-admin-key"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


@pytest.fixture(scope="session")
def client():
    from app.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def admin():
    return {"X-API-Key": "test-admin-key"}
