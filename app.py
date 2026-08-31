from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import tempfile

import config
import utils
import face_extraction
import predict

app = FastAPI(title="Deepfake Detection API")

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "tmp"

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running."}

@app.on_event("startup")
async def startup_event():
    # Load model on startup to fail fast if missing/invalid
    predict.load_model()
    # Ensure temporary upload dir exists
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # 1. Validate File
    try:
        utils.validate_video_extension(file.filename)
        if not file.content_type or not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Save Temporarily
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
            dir=UPLOAD_DIR
        ) as tmp:
            temp_path = tmp.name
            utils.save_upload_file_tmp(file, temp_path)
            print("Info:     Input File Loaded")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    try:        
        # 3. Check Duration
        duration = utils.check_video_duration(temp_path)
        print("Info:     Input File Validated")
        
        # 4. Face Extraction & Preprocessing
        # Returns: (1, SEQUENCE, H, W, C)
        try:
            input_batch = face_extraction.process_video(temp_path)
        except ValueError as e:
            print("Error:    No Face Detected")
            raise HTTPException(status_code=400, detail=str(e)) # No face, etc.

        print("Info:     Face Extraction Completed")
        # 5. Inference
        result = predict.run_inference(input_batch)
        print("Info:    Detection Completed")

        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        # Catch-all for other errors (OpenCV, TF, etc.)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error during processing.")
        
    finally:
        # 6. Cleanup
        if temp_path:
            utils.cleanup_tmp_file(temp_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
