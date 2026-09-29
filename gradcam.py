import base64

import cv2
import numpy as np
import torch


def _get_layer(module: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    current: object = module
    for part in layer_name.split("."):
        if isinstance(current, torch.nn.ModuleDict):
            if part not in current:
                raise ValueError(f"Grad-CAM layer {layer_name!r} was not found.")
            current = current[part]
        elif isinstance(current, torch.nn.Sequential):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                raise ValueError(
                    f"Grad-CAM layer {layer_name!r} was not found."
                ) from None
        elif isinstance(current, torch.nn.Module):
            if not hasattr(current, part):
                raise ValueError(f"Grad-CAM layer {layer_name!r} was not found.")
            current = getattr(current, part)
        else:
            raise ValueError(f"Grad-CAM layer {layer_name!r} was not found.")

    if not isinstance(current, torch.nn.Module):
        raise ValueError(f"Grad-CAM layer {layer_name!r} is not a module.")
    return current


def _encode_overlay(image_rgb: np.ndarray, heatmap: torch.Tensor) -> str:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Grad-CAM images must have RGB shape (height, width, 3).")

    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    heatmap_array = heatmap.detach().cpu().numpy()
    heatmap_array = np.clip(heatmap_array, 0.0, 1.0)
    heatmap_image = np.uint8(heatmap_array * 255)
    heatmap_image = cv2.resize(
        heatmap_image,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_image, cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(image_bgr, 0.6, colored_heatmap_bgr, 0.4, 0)
    success, encoded = cv2.imencode(
        ".jpg",
        overlay_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    if not success:
        raise RuntimeError("Could not encode the Grad-CAM overlay as JPEG.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def generate_sequence_explanation(
    sequence_tensor: torch.Tensor,
    original_frames: list[np.ndarray],
    model: torch.nn.Module,
    target_layer_name: str,
) -> dict:
    """Generate attention-ranked, Base64-encoded Grad-CAM overlays."""
    print(
        f"Info:     Grad-CAM started for {sequence_tensor.shape[1] if sequence_tensor.ndim > 1 else 0} frame(s)"
    )
    if sequence_tensor.ndim != 5 or sequence_tensor.shape[0] != 1:
        raise ValueError("Grad-CAM expects a sequence tensor shaped (1, T, C, H, W).")
    if sequence_tensor.shape[1] != len(original_frames):
        raise ValueError("The number of source frames must match the input sequence.")

    print(f"Info:     Grad-CAM target layer: {target_layer_name}")
    target_layer = _get_layer(model, target_layer_name)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def save_activation(*hook_args):
        output = hook_args[2]
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    hook = target_layer.register_forward_hook(save_activation)
    try:
        model.zero_grad(set_to_none=True)
        batch_size, sequence_length, channels, height, width = sequence_tensor.shape
        frames = sequence_tensor.detach().clone().requires_grad_(True).reshape(
            batch_size * sequence_length,
            channels,
            height,
            width,
        )
        features = model.cnn(frames)
        features = features.reshape(batch_size, sequence_length, -1)
        lstm_output, _ = model.temporal_head.lstm(features)
        attention_scores = model.temporal_head.attention["score"](lstm_output)
        attention_weights = torch.softmax(attention_scores, dim=1).squeeze(-1)
        context = (attention_weights.unsqueeze(-1) * lstm_output).sum(dim=1)
        logit = model.temporal_head.classifier(context).squeeze(-1)
        logit.sum().backward()
    finally:
        hook.remove()

    print("Info:     Grad-CAM activations and gradients captured")
    if len(activations) != 1 or len(gradients) != 1:
        raise RuntimeError("Grad-CAM did not capture the target layer gradients.")

    activation = activations[0]
    gradient = gradients[0]
    if activation.ndim != 4 or gradient.shape != activation.shape:
        raise RuntimeError("Grad-CAM target layer did not produce image feature maps.")

    channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
    heatmaps = torch.relu((channel_weights * activation).sum(dim=1))
    heatmap_max = heatmaps.flatten(1).amax(dim=1, keepdim=True)
    heatmaps = heatmaps / heatmap_max.clamp_min(torch.finfo(heatmaps.dtype).eps).view(-1, 1, 1)
    gradcam_scores = heatmaps.flatten(1).mean(dim=1).detach().cpu()
    attention = attention_weights[0].detach().cpu()
    contribution = attention * gradcam_scores

    print("Info:     Grad-CAM heatmaps calculated; encoding overlays")
    frame_explanations = []
    for position, frame in enumerate(original_frames):
        frame_explanations.append({
            "position": position,
            "attention_weight": float(attention[position]),
            "gradcam_score": float(gradcam_scores[position]),
            "contribution_score": float(contribution[position]),
            "image": {
                "mime_type": "image/jpeg",
                "data": _encode_overlay(frame, heatmaps[position]),
            },
        })

    selected_position = int(torch.argmax(contribution).item())
    selected = frame_explanations[selected_position]
    print(
        f"Info:     Grad-CAM selected frame position {selected_position} "
        f"(contribution={selected['contribution_score']:.4f})"
    )
    return {
        "selected_frame_position": selected_position,
        "attention_weight": selected["attention_weight"],
        "gradcam_score": selected["gradcam_score"],
        "contribution_score": selected["contribution_score"],
        "image": selected["image"],
        "frames": frame_explanations,
    }
