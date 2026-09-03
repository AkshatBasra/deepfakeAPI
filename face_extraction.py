import cv2
import numpy as np
from retinaface import RetinaFace
import config

def extract_frames(video_path: str):
    """
    Extracts evenly spaced frames, capped at the configured sequence length.
    Returns a list of frames (numpy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        print("Info:     Called Frame Extraction")

        if total_frames <= 0:
            return frames

        sample_count = min(config.SEQUENCE_LENGTH, total_frames)
        sample_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
        print(f"Info:     Sampling {sample_count} of {total_frames} frames")

        for frame_index in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        return frames
    finally:
        cap.release()


def detect_and_crop_face(frame):
    """
    Detects the most prominent face in the frame using RetinaFace.
    Returns the cropped and resized face image, or None if no face found.
    """
    # RetinaFace.detect_faces returns a dictionary where keys are 'face_1', 'face_2', etc.
    # We need to handle potential empty results or errors.
    try:
        resp = RetinaFace.detect_faces(frame)
    except Exception as e:
        print(f"RetinaFace error: {e}")
        return None

    if not resp or isinstance(resp, tuple):
        return None

    # Find the largest face (most prominent)
    max_area = 0
    best_face_area = None

    for key in resp:
        face_info = resp[key]
        facial_area = face_info['facial_area']
        x1, y1, x2, y2 = facial_area

        width = x2 - x1
        height = y2 - y1
        area = width * height

        if area > max_area:
            max_area = area
            best_face_area = facial_area

    if best_face_area is not None:
        x1, y1, x2, y2 = best_face_area

        # Ensure coordinates are within image bounds.
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face_img = frame[y1:y2, x1:x2]

        # Resize to input dimensions.
        try:
            face_resized = cv2.resize(face_img, (config.INPUT_WIDTH, config.INPUT_HEIGHT))
            return face_resized
        except Exception:
            return None

    return None

def process_video(video_path: str):
    """
    Full pipeline: Extract frames -> Detect/Crop Faces -> Build Sequence
    Returns: numpy array of shape (1, SEQUENCE_LENGTH, HEIGHT, WIDTH, 3)
    Or raises ValueError if validation fails.
    """
    print("Info:     Called Face Extraction")
    raw_frames = extract_frames(video_path)

    if not raw_frames:
        raise ValueError("Could not extract any frames from video.")

    print("Info:     Frame Extraction Completed")
    processed_faces = []
    print("Info:     RetinaFace Called")
    for frame in raw_frames:
        face = detect_and_crop_face(frame)
        if face is not None:
            processed_faces.append(face)

    print("Info:     RetinaFace Completed")
    if not processed_faces:
        raise ValueError("No faces detected in the video.")

    num_faces = len(processed_faces)

    if num_faces < config.SEQUENCE_LENGTH:
        raise ValueError(f"Not enough face frames detected. Need {config.SEQUENCE_LENGTH}, found {num_faces}.")

    # Uniform sampling if we have too many frames.
    if num_faces > config.SEQUENCE_LENGTH:
        indices = np.linspace(0, num_faces - 1, config.SEQUENCE_LENGTH, dtype=int)
        final_sequence = [processed_faces[i] for i in indices]
    else:
        final_sequence = processed_faces

    print("Info:     Frame Normalisation Called")
    sequence_array = np.array(final_sequence, dtype=np.float32)
    sequence_array /= 255.0

    batch_input = np.expand_dims(sequence_array, axis=0)
    print("Info:     Face normalisation Completed")
    return batch_input
