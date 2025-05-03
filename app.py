import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
import joblib
import random
import hashlib

app = Flask(__name__)

# --- Load Models Once ---
yolo_model = YOLO("best_chicken.pt")
regression_model = joblib.load("regression_model.pkl")

# --- Helper for color ---
def get_color_from_id(chicken_id):
    hash_value = hash(chicken_id)
    random.seed(hash_value)
    return tuple(random.randint(100, 255) for _ in range(3))

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = yolo_model.predict(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    predictions = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        predicted_weight = regression_model.predict([[area]])[0]
        color = get_color_from_id((x1, y1, x2, y2))

        predictions.append({
            'box': [x1, y1, x2, y2],
            'weight': round(predicted_weight, 2),
            'color': color
        })

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
