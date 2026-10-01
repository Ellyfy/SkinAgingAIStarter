from typing import Dict, Tuple

Box = Tuple[int, int, int, int]

def _clip(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        max(0, min(w, x1)),
        max(0, min(h, y1)),
        max(0, min(w, x2)),
        max(0, min(h, y2)),
    )

def approximate_regions(face_box: Tuple[int, int, int, int]) -> Dict[str, Box]:
    _, _, w, h = face_box
    regions = {
        "forehead":    (int(.18*w), int(.05*h), int(.82*w), int(.28*h)),
        "left_eye":    (int(.08*w), int(.25*h), int(.47*w), int(.48*h)),
        "right_eye":   (int(.53*w), int(.25*h), int(.92*w), int(.48*h)),
        "left_cheek":  (int(.06*w), int(.43*h), int(.43*w), int(.72*h)),
        "right_cheek": (int(.57*w), int(.43*h), int(.94*w), int(.72*h)),
        "nose":        (int(.38*w), int(.32*h), int(.62*w), int(.66*h)),
        "nasolabial":  (int(.25*w), int(.55*h), int(.75*w), int(.82*h)),
        "chin":        (int(.30*w), int(.76*h), int(.70*w), int(.96*h)),
    }
    return {name: _clip(box, w, h) for name, box in regions.items()}
