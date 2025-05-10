import numpy as np
import requests
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import base64

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

if 'FASTAPI_SERVER_URL' in os.environ:
    FASTAPI_SERVER_URL = os.environ['FASTAPI_SERVER_URL']  # New: FastAPI server URL
else:
    FASTAPI_SERVER_URL = "http://127.0.0.1:8000"

FASTAPI_SERVER_URL = os.environ['FASTAPI_SERVER_URL']
print(FASTAPI_SERVER_URL)


# For making requests to FastAPI
def request_fastapi(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"image": encoded_str}

        response = requests.post(f"{FASTAPI_SERVER_URL}/predict", json=payload)
        response.raise_for_status()

        result = response.json()
        print("result: ", result)

        # return predicted_class, probability
        predicted_classes = result.get("predictions", [])
        probabilities = result.get("probabilities", [])

        return predicted_classes, probabilities

    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    preds, probs = None, None
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.instance_path, 'uploads', secure_filename(f.filename)))
        img_path = os.path.join(app.instance_path, 'uploads', secure_filename(f.filename))

        preds, probs = request_fastapi(img_path)
        print("Uploaded image path:", img_path)
        print("FastAPI returned:", preds, probs)

    if preds and probs:
        threshold = 0.5
        # Find the top prediction
        top_idx = probs.index(max(probs))
        top_label = preds[top_idx]
        top_confidence = probs[top_idx]

        # Create left-aligned output
        output_html = """
        <div style="text-align: left;">
            <h3>Prediction Results:</h3>
            <ul style="list-style-type: none; padding-left: 0;">
        """
        # for cls, prob in zip(preds, probs):
        #     output_html += f'<li><strong>{cls}</strong>: {prob:.4f}</li>'
        #
        # output_html += "</ul>"
        # output_html += f'<p><strong>Top Prediction:</strong> {top_label} ({top_confidence:.4f})</p>'
        # output_html += "</div>"
        for cls, prob in zip(preds, probs):
            present = "Present" if prob > threshold else "Absent"
            output_html += f'<li><strong>{cls}</strong>: {prob:.4f} — {present}</li>'
        output_html += "</ul></div>"

        return output_html

    return '<a href="#" class="badge badge-warning">Warning</a>'


@app.route('/test', methods=['GET'])
def test():
    img_path = os.path.join(app.instance_path, 'uploads', 'test_image.jpeg')
    preds, probs = request_fastapi(img_path)
    return str(preds)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
