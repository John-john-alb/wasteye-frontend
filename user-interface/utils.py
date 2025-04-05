import os
import requests

# API endpoint for Wasteye
WASTEYE_API_URL = "https://wasteye-docker-243765311449.us-central1.run.app/predict"

def analyze_image_from_url(image_url):
    """
    Sends the image URL directly to the Wasteye API (if the API supports URL-based input).
    """
    response = requests.get(f"{WASTEYE_API_URL}?image_url={image_url}")

    if response.status_code != 200:
        raise RuntimeError(f"Wasteye API request failed with status {response.status_code}: {response.text}")

    data = response.json()
    detections = data.get("results", [{}])[0].get("detections", [])

    formatted_detections = []
    for det in detections:
        formatted_detections.append({
            "label": det.get("class", "UNKNOWN").upper(),
            "box": det.get("bbox", [])
        })

    return {"detections": formatted_detections}
