import tensorflow as tf
import numpy as np
import config
from gradcam import make_gradcam_heatmap, generate_heatmap_overlay
import cv2
import base64

# Global model variable
model = None

def load_model():
    global model
    if config.DEV_NO_MODEL:
        print("Warn:     DEV_NO_MODEL is enabled; skipping model load.")
        return

    if model is None:
        try:
            print(f"Loading model from {config.MODEL_PATH}...")
            # Using compile=False is safer for inference only, sometimes custom losses cause issues
            model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            # We don't raise here to allow building the app, but inference will fail if model is missing
            model = None

def run_inference(input_batch: np.ndarray):
    """
    Runs prediction on the input batch.
    Expected input shape: (1, SEQUENCE_LENGTH, HEIGHT, WIDTH, 3)
    Returns: dictionary with prediction, confidence, heatmap
    """
    global model
    if model is None:
        load_model()
        if model is None:
            # Dev mode: return mock predictions instead of erroring
            if config.DEV_NO_MODEL:
                import random
                confidence_score = round(random.uniform(0.3, 0.95), 2)
                is_fake = confidence_score >= config.FAKE_THRESHOLD
                return {
                    "prediction": "fake" if is_fake else "real",
                    "confidence": confidence_score,
                    "heatmap": None
                }
            raise RuntimeError("Model could not be loaded. Please check config/paths.")

    # 1. Forward Pass
    try:
        predictions = model.predict(input_batch)
    except Exception as e:
         raise RuntimeError(f"Inference failed: {e}")
         
    # Assuming valid output is a single probability score [0.0 - 1.0]
    # Adjust indexing based on your specific model output shape (e.g., could be [batch, 1] or [batch, 2])
    # Here we assume binary classification with sigmoid activation (single output node)
    
    confidence_score = float(predictions[0][0]) if predictions.shape[-1] == 1 else float(predictions[0][1])
    
    # If 2 classes (Real, Fake) and softmax:
    # confidence_score = float(predictions[0][1]) # probability of class 1 (Fake)
    
    # 2. Decision Logic
    is_fake = confidence_score >= config.FAKE_THRESHOLD
    prediction_label = "fake" if is_fake else "real"
    
    # 3. Grad-CAM (Optional)
    heatmap_b64 = None
    if config.ENABLE_GRADCAM:
        try:
            # We need a 4D input for Grad-CAM (Batch, Height, Width, Channels) usually
            # But here we have 5D input (Batch, Sequence, H, W, C) for Video classification (LSTM/Transformer/3DCNN)
            # Grad-CAM on video models is complex.
            # Simplified approach: Apply Grad-CAM on the feature extractor for the *middle frame* of the sequence
            
            # This implementation assumes the model allows access to intermediate layers compatible with this 5D input
            # If the model is a CNN+LSTM, we might need to target the CNN part specifically.
            # For simplicity in this scaffold, we will skipped complex 5D gradcam or assume the user adapts gradcam.py
            # If the user's model is 2D CNN (frame-by-frame) aggregated, we can do it on one frame.
            
            # Let's try to generate heatmap for the middle frame just to show functionality if applicable
            # We'll take the middle frame from the batch
            mid_idx = config.SEQUENCE_LENGTH // 2
            middle_frame = input_batch[0][mid_idx] # (H, W, 3) normalized
             
            # Rescale to 0-255 for overlay
            original_img = np.uint8(middle_frame * 255)
            
            # NOTE: make_gradcam_heatmap expects a batch of images. 
            # If the model expects a sequence, we can't easily pass a single image unless the model structure supports it.
            # Thus, specifically for this scaffold, we will return null unless the user customizes gradcam.py logic heavily.
            # to match their specific architecture (3D CNN vs TimeDistributed 2D CNN).
            
            # Placeholder for successful generation:
            # heatmap = make_gradcam_heatmap(...) 
            # overlay = generate_heatmap_overlay(original_img, heatmap)
            # _, buffer = cv2.imencode('.png', overlay)
            # heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
            pass

        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            heatmap_b64 = None

    return {
        "prediction": prediction_label,
        "confidence": confidence_score,
        "heatmap": heatmap_b64
    }
