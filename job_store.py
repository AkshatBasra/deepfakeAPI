import base64
import json
import shutil
import time
from pathlib import Path
from typing import Any


class JobStore:
    """Filesystem-backed, single-process store for short-lived prediction jobs."""

    def __init__(self, root: str | Path, retention_seconds: int) -> None:
        self.root = Path(root)
        self.retention_seconds = retention_seconds

    def initialize(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        self._write_json(
            job_id,
            {
                "job_id": job_id,
                "status": "processing",
                "stage": 0,
            },
        )

    def update_stage(self, job_id: str, stage: int) -> None:
        job = self._read_json(self._job_path(job_id))
        job["stage"] = stage
        self._write_json(job_id, job)

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        job_dir = self._job_dir(job_id)
        stored_result = self._store_result(job_dir, result)
        self._write_json(
            job_id,
            {
                "job_id": job_id,
                "status": "completed",
                "stage": 8,
                "result": stored_result,
                "expires_at": time.time() + self.retention_seconds,
            },
        )

    def fail(self, job_id: str, error: str) -> None:
        self._write_json(
            job_id,
            {
                "job_id": job_id,
                "status": "failed",
                "stage": 9,
                "error": error,
                "expires_at": time.time() + self.retention_seconds,
            },
        )

    def get(self, job_id: str) -> dict[str, Any] | None:
        job_path = self._job_path(job_id)
        if not job_path.is_file():
            return None

        job = self._read_json(job_path)
        expires_at = job.get("expires_at")
        if expires_at is not None and expires_at <= time.time():
            self._delete_job(job_id)
            return None

        if job["status"] == "completed":
            job["result"] = self._load_result(self._job_dir(job_id), job["result"])
        return job

    def cleanup_expired(self) -> None:
        now = time.time()
        for job_dir in self.root.iterdir():
            if not job_dir.is_dir():
                continue
            job_path = job_dir / "job.json"
            try:
                job = self._read_json(job_path)
            except (FileNotFoundError, json.JSONDecodeError):
                self._delete_job_dir(job_dir)
                continue
            if job.get("expires_at", float("inf")) <= now:
                self._delete_job_dir(job_dir)

    def _store_result(self, job_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
        result_copy = json.loads(json.dumps(result))
        frame_dir = job_dir / "images"
        frame_dir.mkdir(exist_ok=True)

        heatmap = result_copy.get("heatmap")
        if isinstance(heatmap, dict):
            image = heatmap.get("image")
            if isinstance(image, dict):
                self._store_image(image, frame_dir / "selected.jpg")
            for frame in heatmap.get("frames", []):
                image = frame.get("image")
                if isinstance(image, dict):
                    position = frame.get("position", len(list(frame_dir.iterdir())))
                    self._store_image(image, frame_dir / f"frame-{position}.jpg")

        return result_copy

    @staticmethod
    def _store_image(image: dict[str, Any], path: Path) -> None:
        encoded = image.pop("data", None)
        if encoded is None:
            return
        path.write_bytes(base64.b64decode(encoded))
        image["path"] = path.name

    def _load_result(self, job_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
        result_copy = json.loads(json.dumps(result))
        heatmap = result_copy.get("heatmap")
        if isinstance(heatmap, dict):
            image = heatmap.get("image")
            if isinstance(image, dict):
                self._load_image(image, job_dir / "images")
            for frame in heatmap.get("frames", []):
                image = frame.get("image")
                if isinstance(image, dict):
                    self._load_image(image, job_dir / "images")
        return result_copy

    @staticmethod
    def _load_image(image: dict[str, Any], image_dir: Path) -> None:
        image_name = image.pop("path", None)
        if image_name is None:
            return
        image["data"] = base64.b64encode(
            (image_dir / image_name).read_bytes()
        ).decode("ascii")

    def _write_json(self, job_id: str, value: dict[str, Any]) -> None:
        job_dir = self._job_dir(job_id)
        temporary_path = job_dir / "job.json.tmp"
        temporary_path.write_text(json.dumps(value), encoding="utf-8")
        temporary_path.replace(self._job_path(job_id))

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _job_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _delete_job(self, job_id: str) -> None:
        self._delete_job_dir(self._job_dir(job_id))

    @staticmethod
    def _delete_job_dir(job_dir: Path) -> None:
        if job_dir.exists():
            shutil.rmtree(job_dir)
