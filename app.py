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


app = Flask(__name__)
CORS(app)


yolo_models = {
    "yolov8n-seg": YOLO('yolov8n-seg.pt'),
    "yolov8s-seg": YOLO('yolov8s-seg.pt'),
    "yolov8m-seg": YOLO('yolov8m-seg.pt'),
    "yolov8l-seg": YOLO('yolov8l-seg.pt'),
    "yolov8x-seg": YOLO('yolov8x-seg.pt'),

    "myfoodrepo-best": YOLO("myfoodrepo-best.pt"),
    "food-recognition-v0.4-best": YOLO("food-recognition-v0.4-best.pt"),
    
    "food-recognition-v2.1-yolo-v8-m-best": YOLO("food-recognition-v2.1-yolo-v8-m-best.pt"),

    "food-recognition-v2.1-yv8l-75-768-73": YOLO("food-recognition-v2.1-yv8l-75-768-73.pt"),
    "food-recognition-v2.1-yv8l-75-768-75-best": YOLO("food-recognition-v2.1-yv8l-75-768-75-best.pt"),
    "hub-v0.4-272": YOLO("hub-v0.4-272.pt"),
    "hub-v2.1-272": YOLO("hub-v2.1-272.pt"),
    "runs864": YOLO("runs864.pt"),
    "runs166": YOLO("runs166.pt"),
    "best-v0.4-3-yv8m-100-640-55": YOLO("best-v0.4-3-yv8m-100-640-55.pt"),
    "runs355-v0.4-3-yv8m-100-640": YOLO("runs355-v0.4-3-yv8m-100-640.pt"),
}

def store_prediction(filename_original, filename_server, model_name, prediction, confidence):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO predictions (filename_original, filename_server, model_name, prediction, confidence) VALUES (?, ?, ?, ?, ?)
    ''', (filename_original, filename_server, model_name, prediction, confidence))
    conn.commit()
    conn.close()

def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


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
    if not os.path.exists('predictions.db'):
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename_original TEXT,
            filename_server TEXT,
            model_name TEXT,
            prediction TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
        conn.close()
        print("Created 'predictions.db' database.")

@app.route('/')
def index():
    uploads_exists = os.path.exists('uploads')
    uploads_models_exists = os.path.exists('uploads/models')
    db_exists = os.path.exists('predictions.db')
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
def create_database():
    initialize_db()
    return jsonify({"message": "Database created."})



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

    return jsonify({
        "model": model_name,
        "resized_image": f"segmented_{filename_server}",
        "results": results
    })

# =============================
# =============================ß




def get_db_connection():
    conn = sqlite3.connect('predictions.db')  # Adjust the database name as necessary
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/logs/<int:n1>/<int:n2>', defaults={'n2': 5})
@app.route('/logs/<int:n1>', defaults={'n2': 5})
def get_logs(n1, n2):
    conn = get_db_connection()
    query = 'SELECT * FROM predictions ORDER BY created_at DESC LIMIT ? OFFSET ?'
    logs = conn.execute(query, (n2, n1)).fetchall()
    conn.close()

    logs_list = [dict(log) for log in logs]
    return jsonify(logs_list)


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


