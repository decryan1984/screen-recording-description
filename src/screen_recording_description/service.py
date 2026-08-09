"""Local, single-process FastAPI service for describing screen recordings.

POST /runs queues the videos in a given directory. A background thread runs each
video through the pipeline and writes results under output/runs/. Each run's status
is updated in a status.json in its output dir, so run history survives restarts and
is served (with live metrics) via the API for the service dashboard.
"""

import json
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import (
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_VIDEO_BYTES,
    DEFAULT_VLM,
    OLLAMA_BASE_URL,
    SERVICE_RUNS_DIR,
)
from .pipeline import run_pipeline

# In-memory state
_runs = {}
_runs_lock = threading.Lock()
_work = queue.Queue() # run_ids awaiting processing
_started_at = time.time()
_last_processing_sec = None # processing time of the most recent completed run
_last_memory_mb = None # model footprint for the most recent completed run


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value):
    """Strip CR/LF so a crafted filename or error can't inject log lines."""
    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ")
    return value


def _log(event, **fields):
    """Emit one structured JSON log record per line (captured by docker logs)."""
    record = {"ts": _now_iso(), "event": event}
    record.update({k: _sanitize(v) for k, v in fields.items()})
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _validate_video(path):
    """Return None if *path* is an acceptable video, or else an error string."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return "extension_not_allowed"
    if not os.path.isfile(path):
        return "not_a_file"
    try:
        size = os.path.getsize(path)
    except OSError:
        return "stat_failed"
    if size == 0:
        return "empty_file"
    if size > MAX_VIDEO_BYTES:
        return "file_too_large"
    return None


def _ollama_reachable():
    try:
        return requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2).status_code == 200
    except requests.RequestException:
        return False


# Run queue and background worker
def _enqueue(video_path, analysis_prompt=None):
    # Timestamped guid.
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run = {
        "run_id": run_id,
        "video_path": video_path,
        "status": "queued",
        "submitted_at": _now_iso(),
        "analysis_prompt": analysis_prompt,
    }
    with _runs_lock:
        _runs[run_id] = run
    _work.put(run_id)
    return run_id


def _write_status(run):
    """Persist a run's state to output/runs/<dir>/status.json"""
    rel = run.get("run_dir")
    if not rel:
        return  # not started yet (still queued, no output dir)
    path = os.path.join(SERVICE_RUNS_DIR, rel, "status.json")
    try:
        with open(path, "w") as f:
            json.dump(run, f, indent=2)
    except OSError as exc:
        _log("status_write_failed", run_id=run.get("run_id"), error=str(exc))


def _pipeline_output_stats(run_dir, video_name):
    """Read (video_length_sec, frames_processed, model_memory_mb) from the
    pipeline's <video>.json."""
    try:
        with open(os.path.join(run_dir, f"{video_name}.json")) as f:
            data = json.load(f)
        block = data.get("vlm") if isinstance(data.get("vlm"), dict) else {}
        memory = (block.get("performance") or {}).get("model_memory_mb")
        return data.get("video_length_sec"), data.get("frames_processed"), memory
    except (OSError, ValueError):
        return None, None, None


def _update(run_id, **fields):
    with _runs_lock:
        run = _runs.get(run_id)
        if run:
            run.update(fields)
            snapshot = dict(run) if run.get("run_dir") else None
        else:
            snapshot = None
    if snapshot:
        _write_status(snapshot)  # file I/O outside the lock


def _process(run_id):
    global _last_processing_sec, _last_memory_mb
    with _runs_lock:
        run = _runs.get(run_id)
    if not run or run["status"] != "queued":
        return
    video_path = run["video_path"]
    name = os.path.basename(video_path)
    _log("run_received", run_id=run_id, file=name)

    # run_id is already a timestamped guid, so use it directly as the run dir name.
    run_dir = os.path.join(SERVICE_RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    _update(run_id, status="running", started_at=_now_iso(), run_dir=run_id)
    _log("run_started", run_id=run_id, file=name)

    start = time.perf_counter()
    try:
        run_pipeline(video_path, skip_online=True, run_dir=run_dir, evaluate=False,
                     analysis_prompt=run.get("analysis_prompt"))
    except Exception as exc:
        _update(run_id, status="failed", reason="pipeline_error",
                error=f"{type(exc).__name__}: {exc}", finished_at=_now_iso())
        _log("run_failed", run_id=run_id, file=name, reason="pipeline_error", error=str(exc))
        return

    processing = round(time.perf_counter() - start, 2)
    _last_processing_sec = processing
    video_length, frames_processed, memory_mb = _pipeline_output_stats(run_dir, os.path.splitext(name)[0])
    if memory_mb is not None:
        _last_memory_mb = memory_mb
    _update(run_id, status="succeeded", finished_at=_now_iso(),
            processing_sec=processing, video_length_sec=video_length,
            frames_processed=frames_processed)
    _log("run_completed", run_id=run_id, file=name, processing_sec=processing,
         video_length_sec=video_length, frames_processed=frames_processed)


def _worker_loop():
    _log("worker_started", model=DEFAULT_VLM)
    while True:
        run_id = _work.get()
        try:
            _process(run_id)
        finally:
            _work.task_done()


def _load_history():
    """Rebuild run records from status.json so past runs survive a
    restart (display only, no automatical restart for runs that were aborted/failed)."""
    if not os.path.isdir(SERVICE_RUNS_DIR):
        return
    for name in sorted(os.listdir(SERVICE_RUNS_DIR)):
        path = os.path.join(SERVICE_RUNS_DIR, name, "status.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                run = json.load(f)
        except (OSError, ValueError):
            continue
        # A run left 'running'/'queued' means the previous process died mid-flight.
        if run.get("status") in ("running", "queued"):
            run["status"] = "interrupted"
        if run.get("run_id"):
            _runs[run["run_id"]] = run


@asynccontextmanager
async def lifespan(_app):
    os.makedirs(SERVICE_RUNS_DIR, exist_ok=True)
    _load_history()
    threading.Thread(target=_worker_loop, name="worker", daemon=True).start()
    yield


app = FastAPI(title="Screen Recording Description API", version="0.1.0", lifespan=lifespan)


class RunRequest(BaseModel):
    directory: str = Field(..., description="Directory containing the videos (as seen by the service)")
    filenames: list[str] | None = Field(
        default=None,
        description="Specific filenames within the directory - omit to take every eligible video",
    )
    analysis_prompt: str | None = Field(
        default=None,
        max_length=8000,
        description=(
            "Custom analysis prompt run over the frame timeline instead of the "
            "default user-intent inference (e.g. a compliance or best-practice "
            "check). Use the '{entries}' placeholder to control where the "
            "timeline is inserted; if omitted it is appended. Leave unset to keep "
            "the default intent output."
        ),
    )


@app.post("/runs")
def create_runs(req: RunRequest):
    base = os.path.realpath(req.directory)
    if not os.path.isdir(base):
        raise HTTPException(status_code=400, detail=f"Not a directory: {req.directory}")

    if req.filenames is not None:
        names = req.filenames
        report_skips = True
    else:
        # take every file with an accepted extension.
        try:
            names = sorted(os.listdir(base))
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Cannot read directory: {exc}")
        names = [n for n in names
                 if os.path.splitext(n)[1].lower() in ALLOWED_VIDEO_EXTENSIONS]
        report_skips = False

    accepted, rejected = [], []
    for name in names:
        # Only bare filenames are allowed, reject anything with path segments.
        if not name or name != os.path.basename(name):
            rejected.append({"filename": name, "reason": "invalid_filename"})
            continue
        path = os.path.realpath(os.path.join(base, name))
        if os.path.commonpath([base, path]) != base:
            rejected.append({"filename": name, "reason": "outside_directory"})
            continue
        reason = _validate_video(path)
        if reason:
            if report_skips:
                rejected.append({"filename": name, "reason": reason})
            continue
        accepted.append({"filename": name, "run_id": _enqueue(path, req.analysis_prompt)})

    if not accepted and not rejected:
        raise HTTPException(status_code=400, detail="No eligible videos found")
    return {"accepted": accepted, "rejected": rejected}


@app.get("/runs")
def list_runs():
    """Full run history (newest first), for the dashboard."""
    with _runs_lock:
        runs = list(_runs.values())
    runs.sort(key=lambda r: r.get("submitted_at", ""), reverse=True)
    return {"runs": runs}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    with _runs_lock:
        run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    with _runs_lock:
        counts = {s: 0 for s in ("queued", "running", "succeeded", "failed", "interrupted")}
        for run in _runs.values():
            if run["status"] in counts:
                counts[run["status"]] += 1
    return {
        "uptime_sec": round(time.time() - _started_at, 1),
        "model": DEFAULT_VLM,
        "ollama_reachable": _ollama_reachable(),
        "runs": {
            "received": sum(counts.values()),
            "queued": counts["queued"],
            "running": counts["running"],
            "succeeded": counts["succeeded"],
            "failed": counts["failed"],
            "interrupted": counts["interrupted"],
        },
        "last_processing_sec": _last_processing_sec,
        "last_memory_mb": _last_memory_mb,
    }
