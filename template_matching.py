"""
Template Matching Module for Facial Region & Object Localization.
Computes 2D spatial correlation between a reference template and a target image.
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Tuple


MATCH_METHODS = {
    "Correlation Coefficient Normed (Recommended)": cv2.TM_CCOEFF_NORMED,
    "Normalized Cross-Correlation": cv2.TM_CCORR_NORMED,
    "Normalized Squared Difference": cv2.TM_SQDIFF_NORMED,
    "Standard Correlation Coefficient": cv2.TM_CCOEFF,
}


def run_template_matching(
    target_bgr: np.ndarray,
    template_bgr: np.ndarray,
    method_name: str = "Correlation Coefficient Normed (Recommended)",
    multi_scale: bool = True,
    scale_range: Tuple[float, float, int] = (0.6, 1.4, 9)
) -> Dict[str, Any]:
    """
    Perform template matching with optional multi-scale pyramid search.
    Returns best match coordinates, normalized score, correlation heatmap, and visualization.
    """
    method = MATCH_METHODS.get(method_name, cv2.TM_CCOEFF_NORMED)
    is_sqdiff = "SQDIFF" in method_name or method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)
    
    start_time = time.perf_counter()
    
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    
    th, tw = template_gray.shape[:2]
    H, W = target_gray.shape[:2]
    
    # Check dimensions
    if th > H or tw > W:
        # Resize template to fit within target
        scale_fit = min((H - 10) / th, (W - 10) / tw)
        new_w, new_h = max(int(tw * scale_fit), 10), max(int(th * scale_fit), 10)
        template_gray = cv2.resize(template_gray, (new_w, new_h))
        template_bgr = cv2.resize(template_bgr, (new_w, new_h))
        th, tw = template_gray.shape[:2]

    best_val = float("inf") if is_sqdiff else float("-inf")
    best_loc = (0, 0)
    best_scale = 1.0
    best_w, best_h = tw, th
    best_heatmap = None

    if multi_scale:
        scales = np.linspace(scale_range[0], scale_range[1], scale_range[2])
    else:
        scales = [1.0]

    for scale in scales:
        scaled_w = int(tw * scale)
        scaled_h = int(th * scale)
        if scaled_w >= W - 5 or scaled_h >= H - 5 or scaled_w < 15 or scaled_h < 15:
            continue
            
        resized_tpl = cv2.resize(template_gray, (scaled_w, scaled_h))
        res = cv2.matchTemplate(target_gray, resized_tpl, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if is_sqdiff:
            if min_val < best_val:
                best_val = min_val
                best_loc = min_loc
                best_scale = scale
                best_w, best_h = scaled_w, scaled_h
                best_heatmap = res
        else:
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = scale
                best_w, best_h = scaled_w, scaled_h
                best_heatmap = res

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # If single scale or multi-scale didn't capture heatmap
    if best_heatmap is None:
        best_heatmap = cv2.matchTemplate(target_gray, template_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(best_heatmap)
        if is_sqdiff:
            best_val = min_val
            best_loc = min_loc
        else:
            best_val = max_val
            best_loc = max_loc
        best_w, best_h = tw, th
        best_scale = 1.0

    # Calculate match percentage (0 to 100%)
    if is_sqdiff:
        # SQDIFF_NORMED: 0 is perfect, 1 is terrible
        match_confidence = max(0.0, min(100.0, (1.0 - best_val) * 100.0))
    elif method == cv2.TM_CCOEFF_NORMED:
        # Range -1 to +1
        match_confidence = max(0.0, min(100.0, best_val * 100.0))
    else:
        # Other normalized methods: 0 to 1
        match_confidence = max(0.0, min(100.0, best_val * 100.0))

    # Render bounding box onto target image
    annotated = target_bgr.copy()
    top_left = best_loc
    bottom_right = (top_left[0] + best_w, top_left[1] + best_h)

    # Vibrant cyan / purple frame
    cv2.rectangle(annotated, top_left, bottom_right, (255, 0, 180), 3)
    
    # Corner brackets
    corner_len = min(20, best_w // 4, best_h // 4)
    cv2.line(annotated, top_left, (top_left[0] + corner_len, top_left[1]), (255, 255, 255), 4)
    cv2.line(annotated, top_left, (top_left[0], top_left[1] + corner_len), (255, 255, 255), 4)
    cv2.line(annotated, (bottom_right[0], top_left[1]), (bottom_right[0] - corner_len, top_left[1]), (255, 255, 255), 4)
    cv2.line(annotated, (bottom_right[0], top_left[1]), (bottom_right[0], top_left[1] + corner_len), (255, 255, 255), 4)

    # Label with score
    label_text = f"Template Match: {match_confidence:.1f}% (scale {best_scale:.2f}x)"
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw_text, th_text), _ = cv2.getTextSize(label_text, font, 0.55, 1)
    label_y = max(top_left[1] - 8, th_text + 10)
    cv2.rectangle(annotated, (top_left[0], label_y - th_text - 6), (top_left[0] + tw_text + 12, label_y + 4), (255, 0, 180), -1)
    cv2.putText(annotated, label_text, (top_left[0] + 6, label_y - 2), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Normalize heatmap for visual display (0-255)
    norm_heatmap = cv2.normalize(best_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if is_sqdiff:
        # Invert so higher brightness = better match
        norm_heatmap = 255 - norm_heatmap
    color_heatmap = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_INFERNO)

    matched_crop = target_bgr[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()

    return {
        "annotated_bgr": annotated,
        "heatmap_raw": best_heatmap,
        "heatmap_color_bgr": color_heatmap,
        "matched_crop": matched_crop,
        "match_location": top_left,
        "match_size": (best_w, best_h),
        "best_scale": round(float(best_scale), 2),
        "raw_score": round(float(best_val), 4),
        "confidence_percent": round(float(match_confidence), 2),
        "method_name": method_name,
        "execution_time_ms": round(elapsed_ms, 2)
    }
