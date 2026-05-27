import os
import uuid
import sys
import json
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from detector import TranscriptExtractor

app = FastAPI()

JOBS_DIR = Path("/tmp/jobs")
JOBS_DIR.mkdir(exist_ok=True)


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    else:
        try:
            return obj.item()
        except Exception:
            return str(obj)


def save_job(job_id, data):
    with open(JOBS_DIR / f"{job_id}.json", "w") as f:
        json.dump(data, f)


def load_job(job_id):
    p = JOBS_DIR / f"{job_id}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def run_analysis(job_id, video_path, student_id):
    try:
        save_job(job_id, {"status": "processing"})

        extractor = TranscriptExtractor(
            video_path=video_path,
            student_id=student_id,
        )
        result = extractor.extract()
        safe_result = make_json_safe(result.to_dict())
        save_job(job_id, {"status": "done", "result": safe_result})

    except Exception as e:
        save_job(job_id, {"status": "failed", "error": str(e)})
    finally:
        try:
            Path(video_path).unlink()
        except Exception:
            pass


@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    video: UploadFile,
    student_id: str = Form(default="unknown"),
):
    job_id = str(uuid.uuid4())
    suffix = Path(video.filename).suffix or ".mp4"
    tmp = tempfile.mktemp(suffix=suffix)
    with open(tmp, "wb") as f:
        f.write(await video.read())
    background_tasks.add_task(run_analysis, job_id, tmp, student_id)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = load_job(job_id)
    if not job:
        return {"status": "not_found"}
    return job


@app.get("/health")
def health():
    return {"status": "ok"}
