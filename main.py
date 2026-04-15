# main.py
import os
import uuid
import asyncio
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse

from detector import InterviewCheatingDetector

app = FastAPI()

# In-memory job store (use Redis for production scale)
jobs = {}

def run_analysis(job_id: str, video_path: str, student_id: str):
    try:
        jobs[job_id] = {"status": "processing"}
        detector = InterviewCheatingDetector(
            video_path=video_path,
            student_id=student_id,
            process_every_n_frames=3,
            offscreen_duration_s=8.0,
            object_conf=0.55,
            cooldown_s=15.0,
            dir_min_occurrences=6,
            dir_window_s=35.0,
        )
        result = detector.analyze()
        jobs[job_id] = {"status": "done", "result": result.to_dict()}
    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}
    finally:
        try:
            Path(video_path).unlink()
        except:
            pass

@app.post("/analyze")
async def analyze(
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
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return job

@app.get("/health")
def health():
    return {"status": "ok"}