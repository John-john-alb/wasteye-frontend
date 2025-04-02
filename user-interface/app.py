import streamlit as st
from PIL import Image, ImageDraw
from utils import analyze_image  # Import the analyze_image function from utils

# Set up the page config (including title and layout).
st.set_page_config(
    page_title="Wasteye AI",
    layout="wide",
)

# --- Sidebar Configuration ---
st.sidebar.image("wasteyeai_branding.png", use_column_width=True)
st.sidebar.title("Image/Webcam Config")

# Let the user pick a source (only Image or Webcam).
source_option = st.sidebar.radio(
    "Select Source:",
    ("Image", "Webcam"),
    index=0
)

uploaded_file = None
camera_image = None

if source_option == "Image":
    # Image file uploader
    uploaded_file = st.sidebar.file_uploader(
        "Drag and drop file here",
        type=["png", "jpg", "jpeg"],
    )
elif source_option == "Webcam":
    # Capture image from webcam
    camera_image = st.sidebar.camera_input("Take a picture")

# Create a button for detection
detect_button = st.sidebar.button("Detect Objects")

# --- Main Page Layout ---
st.title("WASTEYE AI - Waste Classification using YOLOv8")
st.write("Use the sidebar to upload an image or capture one from your webcam.")

# Create two columns for displaying images side by side
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Uploaded/Captured Image")
    image_to_process = None
    file_to_analyze = None

    if source_option == "Image":
        if uploaded_file is not None:
            # Open the uploaded image for display
            image_to_process = Image.open(uploaded_file)
            st.image(image_to_process, caption="Uploaded Image", use_column_width=True)
            # Use the original file-like object for analysis
            file_to_analyze = uploaded_file
        else:
            st.write("No image uploaded yet.")
    elif source_option == "Webcam":
        if camera_image is not None:
            # Open the captured image for display
            image_to_process = Image.open(camera_image)
            st.image(image_to_process, caption="Captured Image", use_column_width=True)
            # Use the original file-like object for analysis
            file_to_analyze = camera_image
        else:
            st.write("No webcam image captured yet.")

with col2:
    st.subheader("Detection Results")
    if detect_button and file_to_analyze is not None:
        # Reset the file pointer in case it was read already
        file_to_analyze.seek(0)
        try:
            detections_response = analyze_image(file_to_analyze)
        except Exception as e:
            st.error(f"Error analyzing image: {e}")
            detections_response = None

        if detections_response:
            # Assume detections_response is a dictionary with key "detections"
            detections = detections_response.get("detections", [])
            # Draw bounding boxes on a copy of the image
            result_image = image_to_process.copy()
            draw = ImageDraw.Draw(result_image)

            for det in detections:
                label = det.get("label", "")
                box = det.get("box", [])
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    # Draw bounding box
                    draw.rectangle([(x1, y1), (x2, y2)], outline="lime", width=3)
                    # Write the label above the box
                    text = f"{label}"
                    text_size = draw.textsize(text)
                    draw.rectangle([(x1, y1 - text_size[1]), (x1 + text_size[0], y1)], fill="lime")
                    draw.text((x1, y1 - text_size[1]), text, fill="black")

            st.image(result_image, caption="Detection Result", use_column_width=True)
    else:
        st.write("Detection results will appear here once you provide an image and click 'Detect Objects'.")
