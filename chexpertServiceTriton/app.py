import numpy as np
import os
import base64
import uuid
import tritonclient.http as httpclient
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

TRITON_SERVER_URL = os.environ.get("TRITON_SERVER_URL", "localhost:8000")
MODEL_NAME = os.environ.get("TRITON_MODEL_NAME", "chexpert")


# Request to Triton for batched class prediction
def request_triton(image_path):
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        input_data = np.array([[encoded_str]], dtype=object)

        inputs = [httpclient.InferInput("INPUT_IMAGE", [1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(input_data)

        outputs = [
            httpclient.InferRequestedOutput("CHEXPERT_LABELS", binary_data=False),
            httpclient.InferRequestedOutput("CHEXPERT_PROBS", binary_data=False)
        ]

        results = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

        classes = results.as_numpy("CHEXPERT_LABELS")[0]  # list of strings
        probs = results.as_numpy("CHEXPERT_PROBS")[0]  # list of floats

        return list(classes), list(probs)

    except Exception as e:
        print(f"Triton inference error: {e}")
        return None, None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    preds, probs = None, None
    if 'file' not in request.files:
        return '<div class="alert alert-danger">No file uploaded.</div>'

    f = request.files['file']
    file_ext = os.path.splitext(f.filename)[1]
    unique_id = str(uuid.uuid4())
    unique_filename = f"{unique_id}{file_ext}"
    save_path = os.path.join(app.instance_path, 'uploads', unique_filename)
    f.save(save_path)

    preds, probs = request_triton(save_path)
    print("Triton returned:", preds, probs)

    if preds and probs:
        threshold = 0.5
        output_html = f"""
        <div style="text-align: left;">
            <h3>Prediction Results:</h3>
            <ul style="list-style-type: none; padding-left: 0;">
        """
        for cls, prob in zip(preds, probs):
            present = "Present" if prob > threshold else "Absent"
            output_html += f'<li><strong>{cls}</strong>: {prob:.4f} — {present}</li>'
        output_html += "</ul></div>"

        # Optional feedback form can go here (if needed)

        return output_html

    return '<div class="alert alert-danger">Prediction failed.</div>'


@app.route('/test', methods=['GET'])
def test():
    img_path = os.path.join(app.instance_path, 'uploads', 'test_image.jpeg')
    preds, probs = request_triton(img_path)
    return jsonify({"predictions": preds, "probabilities": probs})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
