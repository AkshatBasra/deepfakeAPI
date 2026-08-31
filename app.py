import asyncio
import os
import tempfile
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import face_extraction
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

UPLOAD_DIR = "tmp"
JOB_STORE = {}


def process_prediction(video_path: str):
    """Run the existing RetinaFace + inference pipeline for a saved video."""
    # 1. Check Duration
    duration = utils.check_video_duration(video_path)
    print("Info:     Input File Validated")

    # 2. Face Extraction & Preprocessing
    try:
        input_batch = face_extraction.process_video(video_path)
    except ValueError as e:
        print("Error:    No Face Detected")
        raise HTTPException(status_code=400, detail=str(e))

    print("Info:     Face Extraction Completed")

    # 3. Inference
    result = predict.run_inference(input_batch)
    print("Info:    Detection Completed")
    return result


async def run_prediction_job(job_id: str, video_path: str):
    job = JOB_STORE.get(job_id)
    if job is None:
        return

    try:
        result = await asyncio.to_thread(process_prediction, video_path)
        job["status"] = "completed"
        job["result"] = result
    except HTTPException as exc:
        job["status"] = "failed"
        job["error"] = exc.detail
    except Exception as exc:
        import traceback
        traceback.print_exc()
        job["status"] = "failed"
        job["error"] = str(exc)
    finally:
        utils.cleanup_tmp_file(video_path)


@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running."}


@app.on_event("startup")
async def startup_event():
    # Load model on startup to fail fast if missing/invalid
    predict.load_model()
    # Ensure temporary upload dir exists
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)


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
    JOB_STORE[job_id] = {"job_id": job_id, "status": "processing"}
    asyncio.create_task(run_prediction_job(job_id, temp_path))

    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "processing"})


@app.get("/predict/{job_id}")
async def get_prediction_status(job_id: str):
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "completed":
        response["result"] = job.get("result")
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")

    return response


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
