import asyncio
import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import face_extraction
import config
from job_store import JobStore
import predict
import utils

app = FastAPI(title="Deepfake Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = config.TMP_DIR
JOB_STORE = JobStore(config.JOB_STORE_DIR, config.JOB_RESULT_TTL_SECONDS)
cleanup_task: asyncio.Task[None] | None = None


def process_prediction(video_path: str, update_stage):
    """Run the phase-2 RetinaFace + CNN + LSTM pipeline for a saved video."""
    # 1. Check Duration
    update_stage(1)
    duration = utils.check_video_duration(video_path)
    if duration > config.MAX_VIDEO_DURATION:
        print("Error:    Input File too long")
        raise HTTPException(
            status_code=400,
            detail=f"Video too long. Max duration is {config.MAX_VIDEO_DURATION} seconds.",
        )
    print("Info:     Input File Validated")

    # 2. Face Extraction & Preprocessing
    try:
        input_batch = face_extraction.process_video(
            video_path,
            update_stage=update_stage,
        )
    except ValueError as e:
        print("Error:    No Face Detected")
        raise HTTPException(status_code=400, detail=str(e))

    print("Info:     Face Extraction Completed")

    # 3. Inference
    result = predict.run_inference(input_batch, update_stage=update_stage)
    print("Info:    Detection Completed")
    return result


async def run_prediction_job(job_id: str, video_path: str):
    def update_stage(stage: int) -> None:
        JOB_STORE.update_stage(job_id, stage)

    try:
        result = await asyncio.to_thread(
            process_prediction,
            video_path,
            update_stage,
        )
        JOB_STORE.complete(job_id, result)
    except HTTPException as exc:
        JOB_STORE.fail(job_id, str(exc.detail))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        JOB_STORE.fail(job_id, str(exc))
    finally:
        utils.cleanup_tmp_file(video_path)


async def cleanup_jobs() -> None:
    while True:
        await asyncio.sleep(config.JOB_CLEANUP_INTERVAL_SECONDS)
        await asyncio.to_thread(JOB_STORE.cleanup_expired)


@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running."}


@app.on_event("startup")
async def startup_event():
    # Load model on startup to fail fast if missing/invalid
    predict.load_model()
    # Start with a clean temporary workspace on every process start.
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    JOB_STORE.initialize()
    global cleanup_task
    cleanup_task = asyncio.create_task(cleanup_jobs())


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        utils.validate_video_extension(file.filename)
        if not file.content_type or not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Invalid file type")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=UPLOAD_DIR) as tmp:
            temp_path = tmp.name
            utils.save_upload_file_tmp(file, temp_path)
            print("Info:     Input File Loaded")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    job_id = str(uuid.uuid4())
    JOB_STORE.create(job_id)
    asyncio.create_task(run_prediction_job(job_id, temp_path))

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "processing", "stage": 0},
    )


@app.get("/predict/{job_id}")
async def get_prediction_status(job_id: str):
    job = await asyncio.to_thread(JOB_STORE.get, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
