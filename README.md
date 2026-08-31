# Deepfake Detection Backend

 This is the FastAPI backend for the Deepfake Video Detection application. It handles video uploads, extracts faces using RetinaFace, and runs inference using a pre-trained TensorFlow model.

 ## Setup

 1.  **Prerequisites**: Python 3.9+ installed.
 2.  **Navigate to backend directory**:
     ```bash
     cd backend
     ```
 3.  **Create Virtual Environment**:
     ```bash
     python -m venv venv
     ```
 4.  **Activate Virtual Environment**:
     - Windows: `venv\Scripts\activate`
     - Mac/Linux: `source venv/bin/activate`
 5.  **Install Dependencies**:
     ```bash
     pip install -r requirements.txt
     ```

 ## Configuration

 All tunable parameters are in `config.py`.

 -   **Model**: Place your trained model file (e.g., `model.h5`) in `backend/model/` and update `MODEL_FILENAME` in `config.py`.
 -   **Parameters**:
     -   `SEQUENCE_LENGTH`: Number of frames to extract (default: 16).
     -   `FRAME_SAMPLE_RATE`: Sample every Nth frame (default: 5).
     -   `FAKE_THRESHOLD`: Probability threshold for "fake" class (default: 0.5).

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
 3.  **Note**: The current Grad-CAM implementation is a placeholder skeleton. You may need to adapt `predict.py` logic to correctly target your specific model architecture (e.g. 5D vs 4D inputs).
