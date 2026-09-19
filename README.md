# Deepfake Detection Backend

This is the FastAPI backend for the Deepfake Video Detection application. It
handles video uploads, extracts faces using RetinaFace, and runs phase-1
inference using the CNN trained by the Kaggle notebook.

Phase 1 is intentionally **CNN-only**: the CNN classifies each aligned face
frame and the API averages the eight frame probabilities. The temporal LSTM
head is reserved for phase 2 and is not part of the current checkpoint or API.

 ## Setup

1. **Prerequisites**: Python 3.9+ installed.
2. **Create Virtual Environment**:
     ```bash
     python -m venv venv
     ```
3. **Activate Virtual Environment**:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. **Install Dependencies**:
     ```bash
     pip install -r requirements.txt
     ```

## Model setup

After phase-1 training, copy the notebook's `best.pt` state-dict export to
`model/best.pt`. The `model/` directory is ignored by Git so checkpoints are
not committed.

The backend loads this model at startup and fails if it is missing or
incompatible. For local pipeline/UI work without a checkpoint only, set
`DEV_NO_MODEL = True` in `config.py` to explicitly enable the demo fallback:

All tunable parameters, including `MODEL_FILENAME`, are in `config.py`.

## Running the Server

```bash
uvicorn app:app --reload
```

 The API will be available at `http://localhost:8000`.

 ## API Usage

 ### POST /predict

 Uploads a video file for analysis.

 -   **Headers**: `Content-Type: multipart/form-data`
 -   **Body**: Form-data with key `file` (Video file: .mp4, .avi, .mov)

**Response (JSON)**:
 ```json
 {
   "prediction": "fake", // or "real"
   "confidence": 0.95,   // float 0.0 - 1.0
   "heatmap": null       // base64 string or null
 }
 ```

## Grad-CAM (Optional)

 To enable Grad-CAM heatmaps:
 1.  Set `ENABLE_GRADCAM = True` in `config.py`.
 2.  Update `GRADCAM_LAYER_NAME` in `config.py` to match the target layer of your model.
3. **Note**: Grad-CAM currently targets one phase-1 CNN frame. It is not
   implemented for the future 5D CNN+LSTM input.
