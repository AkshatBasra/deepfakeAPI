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


class CNNFeatureExtractor(nn.Module):
    """Phase-1 EfficientNet-B0 feature extractor."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,
        )
        self.feature_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TemporalAttentionLSTM(nn.Module):
    """Attention-weighted LSTM head from the phase-2 checkpoint."""

    def __init__(
        self,
        input_size: int = 1280,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.ModuleDict({
            "score": nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            ),
        })
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        lstm_output, _ = self.lstm(features)
        attention_scores = self.attention["score"](lstm_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = (attention_weights * lstm_output).sum(dim=1)
        return self.classifier(context).squeeze(-1)


class CNNAttentionLSTM(nn.Module):
    """EfficientNet-B0 frames followed by the phase-2 attention head."""

    def __init__(
        self,
        cnn: CNNFeatureExtractor,
        temporal_head: TemporalAttentionLSTM,
    ) -> None:
        super().__init__()
        self.cnn = cnn
        self.temporal_head = temporal_head

        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

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
        return self.temporal_head(features)


model: CNNAttentionLSTM | None = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_state_dict(path: str) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    if isinstance(checkpoint, dict):
        return {"model_state_dict": checkpoint}
    raise TypeError(f"Checkpoint at {path} is not a state dictionary.")


def _load_backbone() -> CNNFeatureExtractor:
    checkpoint = _load_state_dict(config.CNN_BACKBONE_PATH)
    backbone_state = checkpoint["model_state_dict"]
    state_dict = {
        key.removeprefix("backbone."): value
        for key, value in backbone_state.items()
        if key.startswith("backbone.")
    }
    if not state_dict:
        raise RuntimeError(
            f"CNN checkpoint {config.CNN_BACKBONE_PATH} has no backbone weights."
        )

    cnn = CNNFeatureExtractor()
    cnn.backbone.load_state_dict(state_dict)
    cnn.eval()
    return cnn


def load_model() -> None:
    global model

    if model is not None:
        return

    try:
        print(f"Info:     Loading CNN backbone from {config.CNN_BACKBONE_PATH}...")
        cnn = _load_backbone()

        print(f"Info:     Loading attention head from {config.MODEL_PATH}...")
        attention_checkpoint = _load_state_dict(config.MODEL_PATH)
        temporal_head = TemporalAttentionLSTM()
        temporal_head.load_state_dict(
            attention_checkpoint["model_state_dict"]
        )

        model = CNNAttentionLSTM(cnn=cnn, temporal_head=temporal_head)
        model.to(device).eval()
        print("Info:     CNN + attention LSTM model loaded successfully.")
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        model = None
        raise RuntimeError(
            f"Could not load CNN + attention LSTM model: {exc}"
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
