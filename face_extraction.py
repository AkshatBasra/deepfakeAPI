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

            frames.append(frame)

        return frames
    finally:
        cap.release()


def detect_and_align_face(image_bgr: np.ndarray, target_size: int = 224, max_rotation_deg: float = 40.0):
    try:
        detections = RetinaFace.detect_faces(image_bgr)
    except Exception as e:
        print(f"RetinaFace error: {e}")
        return None

    if not isinstance(detections, dict) or len(detections) == 0:
        return None

    # Find best face based on 'score'
    best_key = max(detections, key=lambda k: detections[k]["score"])
    face = detections[best_key]
    landmarks = face["landmarks"]

    eye_a = np.array(landmarks["left_eye"], dtype=np.float32)
    eye_b = np.array(landmarks["right_eye"], dtype=np.float32)
    left_eye, right_eye = (eye_a, eye_b) if eye_a[0] <= eye_b[0] else (eye_b, eye_a)

    dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    angle = float(np.degrees(np.arctan2(dy, dx)))

    aligned = image_bgr
    if abs(angle) <= max_rotation_deg:
        eyes_center = tuple(((left_eye + right_eye) / 2).astype(np.float32).tolist())
        rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned = cv2.warpAffine(image_bgr, rot_mat, (image_bgr.shape[1], image_bgr.shape[0]))

    x1, y1, x2, y2 = face["facial_area"]
    w, h = x2 - x1, y2 - y1
    mx, my = int(config.FACE_MARGIN * w), int(config.FACE_MARGIN * h)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(aligned.shape[1], x2 + mx), min(aligned.shape[0], y2 + my)

    crop = aligned[y1:y2, x1:x2]
    if crop.size == 0:
        return None
        
    try:
        return cv2.resize(crop, (target_size, target_size))
    except Exception:
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
        face_bgr = detect_and_align_face(frame, target_size=config.INPUT_WIDTH, max_rotation_deg=config.MAX_ROTATION_DEG)
        if face_bgr is not None:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            processed_faces.append(face_rgb)

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

    return final_sequence
