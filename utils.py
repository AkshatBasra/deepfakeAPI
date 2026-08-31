import os
import shutil
from fastapi import UploadFile, HTTPException
import cv2
import config

def validate_video_extension(filename: str):
    allowed_extensions = {'.mp4', '.avi', '.mov'}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file format. Allowed: {', '.join(allowed_extensions)}"
        )

def save_upload_file_tmp(upload_file: UploadFile, destination: str) -> str:
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    finally:
        upload_file.file.close()

def check_video_duration(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Could not open video, possibly corrupt or invalid format
        raise HTTPException(status_code=400, detail="Could not process video file.")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps <= 0 or frame_count <= 0:
        raise HTTPException(status_code=400, detail="Invalid video metadata.")

    duration = frame_count / fps
    if duration > config.MAX_VIDEO_DURATION:
        raise HTTPException(
            status_code=400, 
            detail=f"Video too long. Max duration is {config.MAX_VIDEO_DURATION} seconds."
        )
    
    return duration

def cleanup_tmp_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Error cleaning up file {path}: {e}")
