import numpy as np
import requests
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import base64
import uuid

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
        file_ext = os.path.splitext(f.filename)[1]
        unique_id = str(uuid.uuid4())
        unique_filename = f"{unique_id}{file_ext}"
        save_path = os.path.join(app.instance_path, 'uploads', unique_filename)
        f.save(save_path)

        preds, probs = request_fastapi(save_path)
        print("Uploaded image path:", save_path)
        print("FastAPI returned:", preds, probs)

    if preds and probs:
        threshold = 0.5
        top_idx = probs.index(max(probs))
        top_label = preds[top_idx]
        top_confidence = probs[top_idx]

        output_html = f"""
        <div style="text-align: left;">
            <h3>Prediction Results:</h3>
            <ul style="list-style-type: none; padding-left: 0;">
        """

        for cls, prob in zip(preds, probs):
            present = "Present" if prob > threshold else "Absent"
            output_html += f'<li><strong>{cls}</strong>: {prob:.4f} — {present}</li>'
        output_html += "</ul></div>"

        # Feedback form
        output_html += f"""
        <form id="feedback-form" method="POST" action="/submit_feedback" enctype="multipart/form-data">
            <h4 class="mt-4">Optional Feedback (select labels for each condition):</h4>
        """

        for i, cls in enumerate(preds):
            output_html += f"""
            <div class="mb-2">
                <label>{cls}:</label>
                <select name="label_{i}" class="form-select" style="width:auto; display:inline-block;">
                    <option value="">--</option>
                    <option value="1">Present</option>
                    <option value="0">Absent</option>
                    <option value="-1">Uncertain</option>
                </select>
            </div>
            """

        # 👇 Hidden metadata and actual image file input
        output_html += f"""
            <input type="hidden" name="filename" value="{unique_filename}">
            <input type="hidden" name="image_id" value="{unique_id}">
            <input type="file" name="image" style="display:none;" />
            <button type="submit" class="btn btn-primary mt-3">Submit Feedback</button>
            <a href="/" class="btn btn-secondary mt-3">Skip Feedback</a>
        </form>
        """

        return output_html

    return '<a href="#" class="badge badge-warning">Warning</a>'


@app.route('/test', methods=['GET'])
def test():
    img_path = os.path.join(app.instance_path, 'uploads', 'test_image.jpeg')
    preds, probs = request_fastapi(img_path)
    return str(preds)


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    labels = []
    for i in range(14):  # Expect 14 classes
        val = request.form.get(f'label_{i}', '')
        labels.append(val if val in ['1', '0', '-1'] else '')

    if all(label == '' for label in labels):
        print("User skipped feedback")
        return redirect('/')

    # If user provided any feedback
    image_id = request.form['image_id']
    filename = request.form['filename']
    image_b64 = request.form['image_b64']

    feedback_payload = {
        'image_id': image_id,
        'labels': ','.join(labels),
        'filename': filename
    }

    files = {
        'image': (filename, request.files['image'].read(), 'application/octet-stream')
    }

    try:
        response = requests.post(f"{FASTAPI_SERVER_URL}/submit-feedback", data=feedback_payload, files=files)
        response.raise_for_status()
        # return "<div class='alert alert-success'>Feedback submitted successfully!</div><a href='/' class='btn btn-secondary mt-3'>Back</a>"
        return jsonify({'message': 'Feedback submitted successfully!'})
    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({'message': 'Failed to submit feedback.'}), 500
        # return "<div class='alert alert-danger'>Failed to submit feedback.</div><a href='/' class='btn btn-secondary mt-3'>Back</a>"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
