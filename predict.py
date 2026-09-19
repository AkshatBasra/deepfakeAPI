import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import config
from gradcam import make_gradcam_heatmap, generate_heatmap_overlay
import cv2
import base64

# Define transforms
eval_transform = A.Compose([
    A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ToTensorV2(),
])

class CNNFeatureExtractor(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)

class BinaryClassifierHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)  # (B,) raw logits

class DeepfakeDetectorPhase1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNFeatureExtractor("efficientnet_b0", pretrained=False)
        self.classifier = BinaryClassifierHead(self.cnn.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.cnn.extract_features(x))

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global model
    if model is None:
        if config.DEV_NO_MODEL:
            print("Warn:     DEV_NO_MODEL is enabled; skipping model load.")
            return

        try:
            print(f"Info:     Loading model from {config.MODEL_PATH}...")
            model = DeepfakeDetectorPhase1()
            # The phase-1 notebook exports model.state_dict() to best.pt.
            state_dict = torch.load(config.MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device).eval()
            print("Info:     Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Could not load phase-1 model from {config.MODEL_PATH}: {e}"
            ) from e

def run_inference(input_frames: list):
    """
    Runs prediction on the input sequence of frames (RGB numpy arrays).
    Returns: dictionary with prediction, confidence, heatmap
    """
    global model
    print(f"Info:     Inference started with {len(input_frames)} frame(s)")
    if model is None:
        load_model()
        if model is None:
            if config.DEV_NO_MODEL:
                import random
                confidence_score = round(random.uniform(0.3, 0.95), 2)
                is_fake = confidence_score >= config.FAKE_THRESHOLD
                result = {
                    "prediction": "fake" if is_fake else "real",
                    "confidence": confidence_score,
                    "heatmap": None
                }
                print(f"Info:     Demo inference result: {result}")
                print("Info:     Sending prediction result")
                return result
            raise RuntimeError("Model is not loaded.")

    # 1. Transform and Batch
    tensors = []
    for frame in input_frames:
        tensor = eval_transform(image=frame)["image"]
        tensors.append(tensor)
    
    batch = torch.stack(tensors).to(device) # Shape: (T, 3, 224, 224)
    print(f"Info:     Inference batch shape: {tuple(batch.shape)}")
    print(f"Info:     Inference device: {device}")

    # 2. Forward Pass
    try:
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits)
            # Phase 1: Aggregate probabilities across all valid frames (mean)
            confidence_score = probs.mean().item()
            print(
                "Info:     Frame logits: "
                f"{[round(float(value), 4) for value in logits.detach().cpu()]}"
            )
            print(
                "Info:     Frame fake probabilities: "
                f"{[round(float(value), 4) for value in probs.detach().cpu()]}"
            )
            print(f"Info:     Mean fake probability: {confidence_score:.4f}")
    except Exception as e:
         raise RuntimeError(f"Inference failed: {e}")
    
    # 3. Decision Logic
    is_fake = confidence_score >= config.FAKE_THRESHOLD
    prediction_label = "fake" if is_fake else "real"
    
    # 4. Grad-CAM (Optional)
    heatmap_b64 = None
    if config.ENABLE_GRADCAM:
        try:
            # Generate heatmap for the middle frame for explanation
            mid_idx = len(input_frames) // 2
            mid_tensor = batch[mid_idx:mid_idx+1]
            # OpenCV wants BGR for visualization/saving
            original_img = cv2.cvtColor(input_frames[mid_idx], cv2.COLOR_RGB2BGR)

            heatmap = make_gradcam_heatmap(mid_tensor, model, config.GRADCAM_LAYER_NAME)
            if heatmap is not None:
                overlay = generate_heatmap_overlay(original_img, heatmap)
                _, buffer = cv2.imencode('.png', overlay)
                heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            heatmap_b64 = None

    result = {
        "prediction": prediction_label,
        "confidence": confidence_score,
        "heatmap": heatmap_b64
    }
    print(f"Info:     Final prediction result: {result}")
    print("Info:     Sending prediction result")
    return result
