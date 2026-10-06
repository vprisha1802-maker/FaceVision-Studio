"""
Viola-Jones Algorithm Module for Face, Eye, and Smile Detection.
Leverages Haar Feature-based Cascades for real-time object detection.
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Any


def get_haar_cascades():
    """Load pre-trained Haar Cascade XML classifiers from OpenCV."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    return face_cascade, eye_cascade, smile_cascade


def draw_styled_bbox(img: np.ndarray, x: int, y: int, w: int, h: int, 
                     color: Tuple[int, int, int] = (0, 255, 136), 
                     label: str = "Face", 
                     thickness: int = 2,
                     corner_len: int = 15):
    """
    Draw a sleek, modern tech-styled bounding box with corner accent brackets
    and a gradient-style label pill.
    """
    # Main rectangle with slight transparency or border
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    
    # Corner accent brackets for a sci-fi/modern tech look
    c_color = (255, 255, 255)
    c_thick = thickness + 1
    # Top-Left
    cv2.line(img, (x, y), (x + corner_len, y), c_color, c_thick)
    cv2.line(img, (x, y), (x, y + corner_len), c_color, c_thick)
    # Top-Right
    cv2.line(img, (x + w, y), (x + w - corner_len, y), c_color, c_thick)
    cv2.line(img, (x + w, y), (x + w, y + corner_len), c_color, c_thick)
    # Bottom-Left
    cv2.line(img, (x, y + h), (x + corner_len, y + h), c_color, c_thick)
    cv2.line(img, (x, y + h), (x, y + h - corner_len), c_color, c_thick)
    # Bottom-Right
    cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), c_color, c_thick)
    cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), c_color, c_thick)

    # Label badge above or inside
    if label:
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thick = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
        label_y = max(y - 8, txt_h + 10)
        # Background badge for label
        cv2.rectangle(img, (x, label_y - txt_h - 6), (x + txt_w + 12, label_y + 4), color, -1)
        # Text label (dark text on bright badge)
        cv2.putText(img, label, (x + 6, label_y - 2), font, font_scale, (10, 15, 25), font_thick, cv2.LINE_AA)


def run_viola_jones(
    image_bgr: np.ndarray,
    scale_factor: float = 1.15,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (40, 40),
    detect_eyes: bool = True,
    detect_smile: bool = True
) -> Dict[str, Any]:
    """
    Run Viola-Jones face detection pipeline on input image.
    Returns annotated image, face crops, metrics, and metadata.
    """
    face_cascade, eye_cascade, smile_cascade = get_haar_cascades()
    
    start_time = time.perf_counter()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization to enhance contrast for Haar feature calculation
    gray_eq = cv2.equalizeHist(gray)
    
    # Face detection using cascading classifiers
    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    annotated = image_bgr.copy()
    face_crops = []
    face_data_list = []
    
    for i, (fx, fy, fw, fh) in enumerate(faces):
        roi_gray = gray_eq[fy:fy + fh, fx:fx + fw]
        roi_color = annotated[fy:fy + fh, fx:fx + fw]
        
        # Crop the face
        face_crop = image_bgr[fy:fy + fh, fx:fx + fw].copy()
        face_crops.append(face_crop)
        
        eyes_detected = 0
        smile_detected = False
        
        # Detect Eyes within Face ROI (upper 60% of face)
        if detect_eyes:
            eye_roi_gray = roi_gray[0:int(fh * 0.65), :]
            eye_roi_color = roi_color[0:int(fh * 0.65), :]
            eyes = eye_cascade.detectMultiScale(
                eye_roi_gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(15, 15)
            )
            eyes_detected = len(eyes)
            for (ex, ey, ew, eh) in eyes[:2]:  # Limit to 2 most prominent
                # Draw cyan circle/rect for eyes
                cv2.rectangle(eye_roi_color, (ex, ey), (ex + ew, ey + eh), (255, 212, 0), 2)
                cv2.circle(eye_roi_color, (ex + ew // 2, ey + eh // 2), 2, (0, 255, 255), -1)

        # Detect Smile within Face ROI (lower 50% of face)
        if detect_smile:
            smile_roi_gray = roi_gray[int(fh * 0.5):, :]
            smile_roi_color = roi_color[int(fh * 0.5):, :]
            smiles = smile_cascade.detectMultiScale(
                smile_roi_gray,
                scaleFactor=1.6,
                minNeighbors=22,
                minSize=(25, 20)
            )
            if len(smiles) > 0:
                smile_detected = True
                sx, sy, sw, sh = smiles[0]
                cv2.rectangle(smile_roi_color, (sx, sy), (sx + sw, sy + sh), (128, 0, 255), 2)
        
        # Draw face bounding box with label
        face_label = f"Face #{i + 1}"
        if smile_detected:
            face_label += " (Smiling)"
        draw_styled_bbox(annotated, fx, fy, fw, fh, color=(0, 255, 136), label=face_label)
        
        face_data_list.append({
            "id": i + 1,
            "box": (int(fx), int(fy), int(fw), int(fh)),
            "eyes_count": eyes_detected,
            "smile_detected": smile_detected,
            "area_pixels": int(fw * fh),
        })

    return {
        "annotated_bgr": annotated,
        "faces_count": len(faces),
        "face_crops": face_crops,
        "face_data": face_data_list,
        "execution_time_ms": round(elapsed_ms, 2),
        "algorithm": "Viola-Jones (Haar Cascades)",
        "scale_factor": scale_factor,
        "min_neighbors": min_neighbors,
    }
