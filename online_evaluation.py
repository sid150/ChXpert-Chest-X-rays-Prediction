import os
import base64
import time
import requests
import subprocess

RCLONE_REMOTE = "chi_tacc:object-persist-project29/production"
LOCAL_SYNC_DIR = "chi_production_data"
INFERENCE_URL = "http://fastapi_server:8000/predict"
DELAY_BETWEEN_IMAGES = 3  # seconds

def sync_data_from_chameleon():
    print("Syncing data from Chameleon Cloud...")
    try:
        subprocess.run(["rclone", "mount", RCLONE_REMOTE, LOCAL_SYNC_DIR], check=True)
        print("Data synced successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to sync data: {e}")
        exit(1)

def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_image(image_path):
    encoded_image = encode_image_to_base64(image_path)
    payload = {"image": encoded_image}
    try:
        response = requests.post(INFERENCE_URL, json=payload)
        response.raise_for_status()
        print(f"Sent {image_path}: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending {image_path}: {e}")

def simulate_patient_stream(patient_dir):
    image_files = sorted(os.listdir(patient_dir))
    for image_file in image_files:
        image_path = os.path.join(patient_dir, image_file)
        send_image(image_path)
        time.sleep(DELAY_BETWEEN_IMAGES)

def run_for_all_patients():
    patient_folders = [
        os.path.join(LOCAL_SYNC_DIR, folder)
        for folder in os.listdir(LOCAL_SYNC_DIR)
        if os.path.isdir(os.path.join(LOCAL_SYNC_DIR, folder))
    ]
    for patient_dir in patient_folders:
        print(f"Starting stream for patient: {os.path.basename(patient_dir)}")
        simulate_patient_stream(patient_dir)

if __name__ == "__main__":
    sync_data_from_chameleon()
    run_for_all_patients()
