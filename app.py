import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
import joblib
import random
import hashlib
import os

app = Flask(__name__)

# --- Load Models Once ---
yolo_model = YOLO("best_chicken.pt")
regression_model = joblib.load("regression_model.pkl")

# --- Helper for color ---
def get_color_from_id(chicken_id):
    hash_value = hash(chicken_id)
    random.seed(hash_value)
    return tuple(random.randint(100, 255) for _ in range(3))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert("RGB")

        results = yolo_model.predict(image, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        predictions = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            weight = regression_model.predict([[area]])[0]
            predictions.append({
                "box": [x1, y1, x2, y2],
                "predicted_weight": round(float(weight), 2)
            })

        return jsonify({"predictions": predictions})

    except Exception as e:
        print("❌ ERROR in /predict:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
