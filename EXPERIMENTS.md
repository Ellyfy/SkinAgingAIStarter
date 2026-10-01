# Research experiments

## E0 — Classical baseline
Current Gabor wrinkle proxy.

## E1 — Learned wrinkle segmentation
Replace the Gabor proxy with a U-Net.
Metrics: Dice, IoU, precision, recall, pixel F1.

## E2 — Regional scoring
Compare global vs region-aware wrinkle measurements.

## E3 — Expression awareness
Compare:
A. image -> wrinkle model
B. image -> SMIRK normalization -> wrinkle model

Hypothesis:
SMIRK-based expression/pose control reduces score variability for the same
person across neutral and expressive images.

## E4 — Reference normalization
Compare features to matched reference subsets only after the reference
population is sufficiently large and traceable.

## E5 — Robustness
Evaluate by skin tone, age band, lighting, camera quality, pose, expression.
