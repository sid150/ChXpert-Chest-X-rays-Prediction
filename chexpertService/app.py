import numpy as np
import requests
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import base64

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

FASTAPI_SERVER_URL = os.environ['FASTAPI_SERVER_URL']  # New: FastAPI server URL


# New! for making requests to FastAPI
def request_fastapi(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"image": encoded_str}

        response = requests.post(f"{FASTAPI_SERVER_URL}/predict", json=payload)
        response.raise_for_status()

        result = response.json()
        predicted_class = result.get("prediction")
        probability = result.get("probability")

        return predicted_class, probability

    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    preds = None
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.instance_path, 'uploads', secure_filename(f.filename)))
        img_path = os.path.join(app.instance_path, 'uploads', secure_filename(f.filename))

        preds, probs = request_fastapi(img_path)
        print("📤 Uploaded image path:", img_path)
        print("✅ FastAPI returned:", preds, probs)

        if preds:
            return f'<button type="button" class="btn btn-info btn-sm">{preds}</button>'

    return '<a href="#" class="badge badge-warning">Warning</a>'


@app.route('/test', methods=['GET'])
def test():
    img_path = os.path.join(app.instance_path, 'uploads', 'test_image.jpeg')
    preds, probs = request_fastapi(img_path)
    return str(preds)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
