import os

SEQUENCE_LENGTH = 8         # Number of frames to extract
INPUT_HEIGHT = 224          # Face image height
INPUT_WIDTH = 224           # Face image width
MAX_VIDEO_DURATION = 45     # Maximum video duration in seconds


FAKE_THRESHOLD = 0.5        # Threshold for "fake" classification

# ========== DEV MODE ==========
# Demo mode is enabled by default until a real model is configured.
DEV_NO_MODEL = True # os.getenv("DEV_NO_MODEL", "True").strip().lower() == "true"

# ========== GRAD-CAM ==========
ENABLE_GRADCAM = False      # Enable/Disable Grad-CAM
# NOTE: Update this layer name to match the last convolutional layer of your specific model
GRADCAM_LAYER_NAME = "conv5_block3_out" 

# ========== PATHS ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
# User must update this filename to match the actual uploaded model file
MODEL_FILENAME = "dummy_model.keras" 
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
