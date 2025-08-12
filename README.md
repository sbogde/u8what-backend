# u8what-backend

Flask API for **u8what** — runs YOLOv8 **instance segmentation** on meal photos and returns
detected food items with masks, confidences, and a saved, annotated image.

- Front-end: https://u8what.netlify.app/
- FE repo: https://github.com/sbogde/u8what-front-end
- This repo: `u8what-backend`

## Features

- `/segment` endpoint: upload an image, choose a model, get JSON results + a saved preview.
- Multiple YOLO models supported via a simple name → path cache.
- CORS-enabled for the Netlify frontend.
- Saves uploads and prediction images under `uploads/`.

## Quick start (local)

```bash
# 1) Python env
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 2) Install deps
pip install flask flask-cors ultralytics pillow opencv-python-headless numpy gunicorn

# 3) Put your model(s)
mkdir -p models
# e.g. copy best.pt from training to: models/best.pt

# 4) Run
export MODEL_PATH=models/best.pt
export ALLOWED_ORIGINS=https://u8what.netlify.app
flask --app app.py run  # or: python app.py
```
