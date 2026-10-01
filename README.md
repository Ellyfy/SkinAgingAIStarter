# Expression-Aware Skin Aging AI — Phase 1

This is the first working scaffold for our AI skin-aging application.

## Phase 1 goal
Given a face image:
1. Check image quality.
2. Detect/crop the face.
3. Divide it into approximate facial skin regions.
4. Produce a baseline wrinkle-density proxy.
5. Return regional measurements as JSON.
6. Keep adapters ready for SMIRK, face parsing, learned wrinkle segmentation,
   and reference-population comparison.

## Architecture
```text
image
  |
quality control
  |
face localization
  |
SMIRK adapter --------------> pose / expression / geometry
  |
face parsing adapter -------> skin / eyes / lips / hair masks
  |
skin region extraction
  |
wrinkle model adapter ------> wrinkle mask
  |
regional scoring
  |
reference index ------------> matched population comparison
  |
recommendation engine
```

## Run
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` and use `POST /analyze`.

## Current status
The current wrinkle detector is a classical Gabor-filter baseline so that the
full pipeline runs before model weights are installed. It is not the final
scientific model and is not a medical diagnosis.

Next:
1. BiSeNet facial parsing
2. learned FFHQ-Wrinkle U-Net
3. SMIRK expression/pose integration
4. reference-population index
5. pigmentation, texture, pores, redness
