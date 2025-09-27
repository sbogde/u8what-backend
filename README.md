# u8what-backend

Flask backend for **u8what**. Accepts food photos, runs YOLOv8 instance segmentation,
persists detections in SQLite, and serves annotated assets for the front end at
https://u8what.food.

- Front end: https://u8what.food (also available via https://u8what.lol and https://u8what.netlify.app)
- Front-end source: https://github.com/sbogde/u8what-front-end
- This repo: `u8what-backend`

## Features

- Accepts uploads on `/segment`, runs the selected YOLOv8 model, stores results, and returns JSON with detections.
- Saves originals and segmented previews under `uploads/`, making them accessible via `/uploads/<filename>`.
- Persists segmentation metadata plus per-detection rows in `u8what.db` for later browsing.
- Exposes `/logs` for a paginated history of recent classifications.
- Provides maintenance helpers (`/mkdirs`, `/mkdb`) to bootstrap storage and database.
- Restricts CORS to trusted clients (`http://localhost` on any port, `https://u8what.food`, `https://u8what.lol`, `https://u8what.netlify.app`).
- Optional NSFW guard using `nsfw-detector` when an NSFW model path is provided.

## Quick start (local)

```bash
# 1) Python env
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 2) Install dependencies
pip install -r requirements.txt  # add nsfw-detector if you plan to enable the safety filter

# 3) Ensure model weights are available (place *.pt files alongside app.py)
# e.g. cp /path/to/yolov8n-seg.pt .

# 4) Run the API
python app.py  # or: flask --app app.py run

# (optional) enable safety filter
export NSFW_MODEL_PATH=/path/to/nsfw_mobilenet2.224x224.h5
export NSFW_THRESHOLD=0.82  # tweak if needed
```

When `FLASK_ENV=production` is set, the app ensures `uploads/` exists and creates the SQLite
schema automatically.

## API overview

- `GET /` – returns a basic health payload with flags for uploads and DB presence.
- `POST /segment` – multipart upload (`image`, optional `model`) that writes segmentation history and returns detections.
- `GET /logs?page=1&page_size=10` – paginated segmentation history from `u8what.db`.
- `POST /mkdirs` – ensures upload directories exist.
- `POST /mkdb` – initializes the SQLite schema in `u8what.db`.
- `GET /uploads/<filename>` – serves stored originals or annotated previews.
