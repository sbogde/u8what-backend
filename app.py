# app.py
import numpy as np
import os
import sqlite3
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
# from waitress import serve


from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input # type: ignore


from ultralytics import YOLO

# --- DB config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "u8what.db")

app = Flask(__name__)
CORS(app)


yolo_models = {
    "yolov8n-seg": YOLO('yolov8n-seg.pt'),
    "yolov8s-seg": YOLO('yolov8s-seg.pt'),
    "yolov8m-seg": YOLO('yolov8m-seg.pt'),
    "yolov8l-seg": YOLO('yolov8l-seg.pt'),
    "yolov8x-seg": YOLO('yolov8x-seg.pt'),


    "v0.4-Ultralytics-Hub": YOLO("hub-v0.4-272.pt"),
    "v2.1-Ultralytics-Hub": YOLO("hub-v2.1-272.pt"),

    "v0.4-Google-Colab": YOLO("runs355-v0.4-3-yv8m-100-640.pt"),
    "v2.1-Google-Colab": YOLO("runs061-v2.1-4-yv8l-75-768.pt"),

    "v0.4-Mici-Google-Colab": YOLO("mici-v0.4-yv8m-100-640.pt"),
    "v0.4_mici_plus3_stageB_epoch_02": YOLO("v0.4_mici_plus3_stageB_epoch_02.pt"),
    "v0.4_mici_sarmale_mamaliga": YOLO("v0.4_mici_sarmale_mamaliga.pt"),

    "v2.1_plus_yorkshire_pudding_gc": YOLO("v2.1_plus_yorkshire_pudding_gc.pt"),
    "v2.1_plus_yorkshire_pudding_uhub": YOLO("v2.1_plus_yorkshire_pudding_uhub.pt"),
}

def _pct(x):
    # normalise confidences into 0–100 (your model sometimes returns 0..1, sometimes 0..100)
    return float(x) * 100.0 if 0.0 <= x <= 1.0 else float(x)

def _labels_summary(results):
    from collections import Counter
    c = Counter(r["label"] for r in results)
    import json
    return json.dumps(c, ensure_ascii=False)


# def preprocess_image(img_path, target_size=(224, 224)):
#     img = load_img(img_path, target_size=target_size)
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)
#     return img


def denormalize_image(img_array):
    # Convert to float32 for safe addition
    img_array = img_array.astype(np.float32)

    # Add your mean values
    img_array += [123.68, 116.779, 103.939]

    # Clip and convert back to uint8
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return img_array

def save_image(img_array, save_path):
    img_array = denormalize_image(img_array)
    img_array = np.uint8(img_array)
    img = Image.fromarray(img_array)
    img.save(save_path)


def initialize_db():
    schema = """
    CREATE TABLE IF NOT EXISTS segmentations (
      id               INTEGER PRIMARY KEY AUTOINCREMENT,
      filename_original TEXT,
      filename_server   TEXT,
      resized_image     TEXT,
      model_name        TEXT NOT NULL,
      num_detections    INTEGER NOT NULL DEFAULT 0,
      top_label         TEXT,
      top_confidence    REAL,
      labels_json       TEXT,
      created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS detections (
      id               INTEGER PRIMARY KEY AUTOINCREMENT,
      segmentation_id  INTEGER NOT NULL REFERENCES segmentations(id) ON DELETE CASCADE,
      label            TEXT NOT NULL,
      confidence       REAL NOT NULL,
      idx              INTEGER,
      UNIQUE(segmentation_id, idx) ON CONFLICT IGNORE
    );

    CREATE INDEX IF NOT EXISTS idx_segmentations_created ON segmentations(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_segmentations_model   ON segmentations(model_name);
    CREATE INDEX IF NOT EXISTS idx_detections_seg        ON detections(segmentation_id);
    CREATE INDEX IF NOT EXISTS idx_detections_label      ON detections(label);
    """
    with get_db_connection() as conn:
        conn.executescript(schema)
    print("Created the database.")

@app.route('/')
def index():
    uploads_exists = os.path.exists('uploads')
    uploads_models_exists = os.path.exists('uploads/models')
    db_exists = os.path.exists(DB_PATH)
    return jsonify({
        "message": "Welcome to the Image Classification API!",
        "uploads_exists": uploads_exists,
        "uploads_models_exists": uploads_models_exists,
        "db_exists": db_exists
    })

@app.route('/mkdirs', methods=['POST'])
def create_directories():
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('uploads/models', exist_ok=True)
    return jsonify({"message": "Directories created."})

@app.route('/mkdb', methods=['POST'])
def mkdb():
    initialize_db()
    return jsonify({"ok": True, "db": os.path.basename(DB_PATH)})



@app.route('/segment', methods=['POST'])
def segment_food_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files['image']
    filename_original = file.filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename_server = f"{timestamp}_{filename_original}"
    filepath = os.path.join('uploads', filename_server)
    file.save(filepath)

    model_name = request.form.get('model', 'yolov8n-seg')
    model = yolo_models.get(model_name, yolo_models['yolov8n-seg'])

    results = model(filepath)
    result = results[0]  # Grab the first (and only) result

    # Save the annotated image
    annotated_img_path = os.path.join('uploads/models', f"segmented_{filename_server}")
    result.save(filename=annotated_img_path)

 # Extract predictions manually
    results = []
    if result.boxes is not None:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            results.append({
                'label': result.names[int(cls)],
                'confidence': float(conf) * 100
            })

    # Compute summary
    if results:
        top = max(results, key=lambda r: _pct(r.get("confidence", 0)))
        top_label = top.get("label")
        top_conf  = _pct(top.get("confidence", 0))
    else:
        top_label = None
        top_conf  = None

    labels_json = _labels_summary(results)
    num_detections = len(results)

    with get_db_connection() as conn:
        cur = conn.cursor()
        # 1) insert parent row
        cur.execute("""
            INSERT INTO segmentations
            (filename_original, filename_server, resized_image, model_name,
            num_detections, top_label, top_confidence, labels_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (filename_original, filename_server, f"segmented_{filename_server}", model_name,
            num_detections, top_label, top_conf, labels_json))
        seg_id = cur.lastrowid

        # 2) insert children
        cur.executemany("""
            INSERT INTO detections (segmentation_id, label, confidence, idx)
            VALUES (?, ?, ?, ?)
        """, [
            (seg_id, r.get("label"), _pct(r.get("confidence", 0)), i)
            for i, r in enumerate(results)
        ])


    return jsonify({
        "model": model_name,
        "resized_image": f"segmented_{filename_server}",
        "results": results
    })

# =============================
# =============================ß

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    # Hardening / performance:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")  # ms
    return conn

@app.route('/logs', methods=['GET'])
def logs():
    # query params: ?page=1&page_size=20
    try:
        page      = max(int(request.args.get('page', 1)), 1)
        page_size = min(max(int(request.args.get('page_size', 10)), 1), 100)
    except ValueError:
        page, page_size = 1, 10
    offset = (page - 1) * page_size

    with get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM segmentations").fetchone()[0]
        rows  = conn.execute("""
            SELECT id, filename_server, resized_image, model_name,
                   top_label, top_confidence, num_detections, created_at
            FROM segmentations
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
        """, (page_size, offset)).fetchall()

    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "image": r["resized_image"],
            "filename_server": r["filename_server"],
            "model": r["model_name"],
            "prediction": r["top_label"],
            "confidence": r["top_confidence"],   # send raw; format as "71.33%" in UI
            "num_detections": r["num_detections"],
            "date": r["created_at"]
        })

    return jsonify({
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": items
    })
# @app.route('/logs/<int:n1>/<int:n2>', defaults={'n2': 5})
# @app.route('/logs/<int:n1>', defaults={'n2': 5})
# def get_logs(n1, n2):
#     conn = get_db_connection()
#     query = 'SELECT * FROM predictions ORDER BY created_at DESC LIMIT ? OFFSET ?'
#     logs = conn.execute(query, (n2, n1)).fetchall()
#     conn.close()

#     logs_list = [dict(log) for log in logs]
#     return jsonify(logs_list)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Ensure the uploads directories and database are created
if os.environ.get('FLASK_ENV') == 'production':
    # Ensure the uploads directories and database are created
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('uploads/models', exist_ok=True)
    print("Created 'uploads' directory.")
    print("Created 'uploads/models' directory.")
    initialize_db()


if __name__ == '__main__':
    # Ensuring the directories are created
    # if not os.path.exists('uploads'):
    #     os.makedirs('uploads')
    #     print("Created 'uploads' directory.")
    # if not os.path.exists('uploads/models'):
    #     os.makedirs('uploads/models')
    #     print("Created 'uploads/models' directory.")

    app.run(debug=True)
    # app.run(host="0.0.0.0", port=5000, debug=True)

    # if os.environ.get('FLASK_ENV') == 'production':
        # serve(app, host='0.0.0.0', port=5000)
    # else:
        # app.run(host="0.0.0.0", port=5000, debug=True)


