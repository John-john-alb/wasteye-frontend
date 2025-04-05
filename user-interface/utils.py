import os
import requests

# API endpoint for Wasteye
WASTEYE_API_URL = "https://wasteye-243765311449.europe-west1.run.app/predict"

def analyze_image(file_like):
    """
    Analyzes an image file using the Wasteye API.

    Args:
        file_like (file-like object): An image file object (like Streamlit's UploadedFile).

    Returns:
        dict: Parsed JSON response with detections.
    """
    files = {"file": ("uploaded_image.jpg", file_like, "image/jpeg")}
    response = requests.post(WASTEYE_API_URL, files=files)

    if response.status_code != 200:
        raise RuntimeError(f"Wasteye API request failed with status {response.status_code}: {response.text}")

    return response.json()
