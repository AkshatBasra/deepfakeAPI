import os

SEQUENCE_LENGTH = 8         # Number of frames to extract
INPUT_HEIGHT = 224          # Face image height
INPUT_WIDTH = 224           # Face image width
MAX_VIDEO_DURATION = 45     # Maximum video duration in seconds
FACE_MARGIN = 0.15          # 15% margin for face cropping
MAX_ROTATION_DEG = 40.0     # Maximum face rotation in degrees
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


FAKE_THRESHOLD = 0.5        # Threshold for "fake" classification

# ========== DEV MODE ==========
# Keep demo mode disabled for normal operation. Enable only for local UI testing.
DEV_NO_MODEL = False

# ========== GRAD-CAM ==========
ENABLE_GRADCAM = False      # Enable/Disable Grad-CAM
# NOTE: Update this layer name to match the last convolutional layer of your specific model
GRADCAM_LAYER_NAME = "cnn.backbone.conv_head" 

# ========== PATHS ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
# The phase-1 notebook writes this state dict as best.pt.
MODEL_FILENAME = "best.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
