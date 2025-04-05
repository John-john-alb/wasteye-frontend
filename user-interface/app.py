import requests
from io import BytesIO
import streamlit as st
from PIL import Image, ImageDraw
from utils import analyze_image_from_url  # Import the analyze_image function from utils

# --- Streamlit UI ---
st.set_page_config(page_title="Wasteye AI", layout="wide")

# Sidebar
st.sidebar.image("wasteyeai_branding.png", use_column_width=True)
st.sidebar.title("Image from URL")

image_url = st.sidebar.text_input("Enter image URL")
detect_button = st.sidebar.button("Detect Objects")

# Main layout
st.title("WASTEYE AI - Waste Classification using YOLOv8")
st.write("Paste an image URL in the sidebar and click 'Detect Objects'.")

col1, col2 = st.columns(2, gap="large")

image_to_process = None

with col1:
    st.subheader("Image from URL")
    if image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_to_process = Image.open(BytesIO(response.content))
            st.image(image_to_process, caption="Image from URL", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to load image from URL: {e}")
    else:
        st.write("No image URL provided.")

with col2:
    st.subheader("Detection Results")
    if detect_button and image_to_process:
        try:
            detections_response = analyze_image_from_url(image_url)
        except Exception as e:
            st.error(f"Error analyzing image: {e}")
            detections_response = None

        if detections_response:
            detections = detections_response.get("detections", [])
            result_image = image_to_process.copy()
            draw = ImageDraw.Draw(result_image)

            for det in detections:
                label = det.get("label", "")
                box = det.get("box", [])
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    draw.rectangle([(x1, y1), (x2, y2)], outline="lime", width=3)
                    text = f"{label}"
                    text_size = draw.textsize(text)
                    draw.rectangle([(x1, y1 - text_size[1]), (x1 + text_size[0], y1)], fill="lime")
                    draw.text((x1, y1 - text_size[1]), text, fill="black")

            st.image(result_image, caption="Detection Result", use_column_width=True)
    else:
        st.write("Detection results will appear here once you provide an image URL and click 'Detect Objects'.")
