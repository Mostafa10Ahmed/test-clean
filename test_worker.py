import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import worker_server

# CLI مزيف بينسخ الملفات من -i لـ -o (نفس فكرة اختبارات cli_cleaner)
FAKE_CLI = """
import pathlib, shutil, sys
args = sys.argv[1:]
src = pathlib.Path(args[args.index('-i') + 1])
dst = pathlib.Path(args[args.index('-o') + 1]) / src.name
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.is_file():
        shutil.copy(f, dst / f.name)
"""


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "manga_translator").mkdir(parents=True)
    (repo / "manga_translator" / "__init__.py").write_text("")
    (repo / "manga_translator" / "__main__.py").write_text(FAKE_CLI)
    monkeypatch.setattr(worker_server, "WORKER_TOKEN", "secret")
    monkeypatch.setattr(worker_server, "TRANSLATOR_REPO", str(repo))
    monkeypatch.setattr(worker_server, "WORK_ROOT", str(tmp_path / "work"))
    return TestClient(worker_server.app)


def make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_rejects_wrong_token(client):
    resp = client.post(
        "/clean",
        files={"file": ("c.zip", make_zip({"1.png": b"a"}), "application/zip")},
        data={"profile": "fast"},
        headers={"X-Worker-Token": "wrong"},
    )
    assert resp.status_code == 401


def test_rejects_unknown_profile(client):
    resp = client.post(
        "/clean",
        files={"file": ("c.zip", make_zip({"1.png": b"a"}), "application/zip")},
        data={"profile": "turbo"},
        headers={"X-Worker-Token": "secret"},
    )
    assert resp.status_code == 400


def test_returns_cleaned_zip(client):
    resp = client.post(
        "/clean",
        files={"file": ("c.zip", make_zip({"2.png": b"two", "1.png": b"one"}), "application/zip")},
        data={"profile": "fast"},
        headers={"X-Worker-Token": "secret"},
    )
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        assert zf.namelist() == ["1.png", "2.png"]
        assert zf.read("2.png") == b"two"
