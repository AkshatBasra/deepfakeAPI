import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


eval_transform = A.Compose([
    A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ToTensorV2(),
])


class CNNLSTMHybrid(nn.Module):
    """EfficientNet-B0 frame features followed by temporal classification."""

    def __init__(
        self,
        cnn: nn.Module,
        input_size: int = 1280,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = cnn

        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = x.shape
        frames = x.reshape(
            batch_size * sequence_length,
            channels,
            height,
            width,
        )

        with torch.no_grad():
            features = self.cnn(frames)

        features = features.reshape(batch_size, sequence_length, -1)
        lstm_output, _ = self.lstm(features)
        last_output = lstm_output[:, -1, :]
        return self.classifier(last_output).squeeze(-1)


model: CNNLSTMHybrid | None = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> None:
    global model

    if model is not None:
        return

    try:
        print(f"Info:     Loading model from {config.MODEL_PATH}...")
        cnn = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,
        )
        checkpoint = torch.load(config.MODEL_PATH, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            input_size = checkpoint.get("input_size", 1280)
            hidden_size = checkpoint.get("hidden_size", 256)
            num_layers = checkpoint.get("num_layers", 2)
            dropout = checkpoint.get("dropout", 0.3)
        else:
            state_dict = checkpoint
            input_size = 1280
            hidden_size = 256
            num_layers = 2
            dropout = 0.3

        model = CNNLSTMHybrid(
            cnn=cnn,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.load_state_dict(state_dict)
        model.to(device).eval()
        print("Info:     CNN + LSTM model loaded successfully.")
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        model = None
        raise RuntimeError(
            f"Could not load CNN + LSTM model from {config.MODEL_PATH}: {exc}"
        ) from exc


def run_inference(input_frames: list) -> dict:
    """
    Predict whether an eight-frame RGB face sequence is fake or real.

    Returns a video-level fake probability and prediction label.
    """
    global model

    print(f"Info:     Inference started with {len(input_frames)} frame(s)")
    if len(input_frames) != config.SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected {config.SEQUENCE_LENGTH} face frames, "
            f"received {len(input_frames)}."
        )

    if model is None:
        load_model()
        if model is None:
            raise RuntimeError("Model is not loaded.")

    tensors = [
        eval_transform(image=frame)["image"]
        for frame in input_frames
    ]
    sequence = torch.stack(tensors).unsqueeze(0).to(device)

    print(f"Info:     Inference input shape: {tuple(sequence.shape)}")
    print(f"Info:     Inference device: {device}")

    try:
        with torch.no_grad():
            logit = model(sequence)
            confidence_score = torch.sigmoid(logit).item()
    except (RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc

    print(f"Info:     Video fake probability: {confidence_score:.4f}")

    is_fake = confidence_score >= config.FAKE_THRESHOLD
    prediction_label = "fake" if is_fake else "real"
    result = {
        "prediction": prediction_label,
        "confidence": confidence_score,
        "heatmap": None,
    }
    print(f"Info:     Final prediction result: {result}")
    print("Info:     Sending prediction result")
    return result
