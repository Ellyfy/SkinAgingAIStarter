import cv2
import numpy as np

def wrinkle_proxy_mask(face_bgr: np.ndarray) -> np.ndarray:
    """Classical Gabor baseline; not the final learned wrinkle model."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    responses = []
    for theta in np.linspace(0, np.pi, 8, endpoint=False):
        kernel = cv2.getGaborKernel(
            (17, 17), 3.2, float(theta), 8.0, 0.5, 0, ktype=cv2.CV_32F
        )
        responses.append(
            np.abs(cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel))
        )

    response = np.max(np.stack(responses), axis=0)
    threshold = float(np.percentile(response, 93.0))
    mask = (response >= threshold).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

def score_regions(mask, regions):
    h, w = mask.shape[:2]
    full_area = max(1, h * w)
    out = {}

    for name, (x1, y1, x2, y2) in regions.items():
        roi = mask[y1:y2, x1:x2]
        area = int(roi.size)
        out[name] = {
            "wrinkle_density": round(float(roi.mean()) if area else 0.0, 4),
            "relative_area": round(area / full_area, 4),
        }
    return out

def overall_score(region_scores):
    if not region_scores:
        return 0.0
    return round(
        float(np.mean([v["wrinkle_density"] for v in region_scores.values()])),
        4,
    )
