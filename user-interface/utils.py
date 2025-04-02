import os
import json
import requests
import base64
from openai import OpenAI
from PIL import Image, ImageDraw
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# Read API key from environment variable
import os
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

# System prompt for image analysis
system_prompt = '''
Identify different types of waste in an image and classify them using predefined categories, while providing bounding boxes for each identified waste.

Input an image of household waste, and the system will recognize individual waste items within the image. Each item will be classified according to the specified categories and its location will be represented as a bounding box.

- **Classifications**:
- Biodegradable
- Cardboard
- Metal
- Paper
- Glass
- Plastic

- **Bounding Box Representation**: Each waste item should include a bounding box indicating its location in the image. The bounding box should be presented as a list of coordinates representing [x_min, y_min, x_max, y_max].

# Output Format

```json
{
"detections": [
    {"label": "PLASTIC", "box": [50, 50, 200, 200]},
    {"label": "METAL", "box": [220, 80, 330, 190]}
]
}
'''

# Function to encode image-like objects to base64
def encode_image_bytes(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

# Main analyze function
def analyze_image(image_input):
    """
    Accepts an image file-like object (from uploader or webcam) or a string URL.
    Sends the image to OpenAI for waste detection.
    """

    # Image from uploaded file (file-like object)
    base64_image = encode_image_bytes(image_input)
    image_payload = {
        "url": f"data:image/jpeg;base64,{base64_image}"
        }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image_payload}
                ]
            }
        ],
        max_tokens=300,
        top_p=0.1
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "").strip("` \n")
    elif content.startswith("```"):
        content = content.strip("` \n")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse GPT response:\n{content}")
