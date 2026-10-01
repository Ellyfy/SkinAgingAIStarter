import numpy as np
from app.scoring import score_regions, overall_score

def test_region_scoring():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :50] = 1
    regions = {
        "top_left": (0, 0, 50, 50),
        "bottom_right": (50, 50, 100, 100),
    }
    scores = score_regions(mask, regions)
    assert scores["top_left"]["wrinkle_density"] == 1.0
    assert scores["bottom_right"]["wrinkle_density"] == 0.0
    assert overall_score(scores) == 0.5
