# src/demo_app.py
import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont # Import ImageDraw and ImageFont
import io
from collections import defaultdict # Might be useful for counts

# Configuration
API_SERVICE_NAME = "vegetable-counter-api" # This must match the service name in your docker-compose.yml
API_PORT = 8000
API_URL = f"http://{API_SERVICE_NAME}:{API_PORT}/predict"

st.set_page_config(layout="wide") # Use wide layout for better image display
st.title("🌱 Intelligent Inventory Counter Demo")
st.markdown("Upload an image of vegetables, and the model will detect and count them!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image_bytes = uploaded_file.getvalue()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    st.sidebar.image(original_image, caption="Original Image", use_column_width=True) # Display original in sidebar

    st.write("### Detecting items...")

    # Prepare image for API request
    files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
    
    try:
        response = requests.post(API_URL, files=files)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        detections = response.json()

        st.subheader("Results:")
        
        # Check if predictions were successful
        if detections and detections.get("predictions"):
            # Create a drawable image copy to draw on
            draw_image = original_image.copy()
            draw = ImageDraw.Draw(draw_image)
            
            # Load a font for text (adjust path as needed, or use default)
            try:
                font = ImageFont.truetype("arial.ttf", 20) # Try a common font
            except IOError:
                font = ImageFont.load_default() # Fallback to default font if not found

            # Loop through predictions and draw bounding boxes
            detected_counts = defaultdict(int)
            for det in detections["predictions"]:
                box_coords = det["bounding_box"] # x1, y1, x2, y2
                class_name = det["class_name"]
                confidence = det["confidence"]
                
                # Draw rectangle
                draw.rectangle([box_coords["x1"], box_coords["y1"], box_coords["x2"], box_coords["y2"]], 
                               outline="red", width=3) # Use x1,y1,x2,y2

                # Draw label text
                label_text = f"{class_name} ({confidence:.2f})"
                text_x = box_coords["x1"]
                text_y = box_coords["y1"] - 25 # Place text above the box

                # Add a background rectangle for text for better readability
                text_bbox = draw.textbbox((text_x, text_y), label_text, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((text_x, text_y), label_text, fill="white", font=font)
                
                detected_counts[class_name] += 1

            st.image(draw_image, caption="Detected Items", use_column_width=True)
            
            st.write("### Item Counts:")
            if detected_counts:
                for item, count in detected_counts.items():
                    st.write(f"- **{item.capitalize()}**: {count}")
                st.write(f"---")
                st.write(f"**Total Detected Items**: {sum(detected_counts.values())}")
            else:
                st.write("No items detected.")

        else:
            st.write("No predictions found in the image.")
            # If the API returns valid JSON but no predictions array or it's empty
            st.json(detections) # Still show raw JSON for debugging


    except requests.exceptions.ConnectionError:
        st.error("API service is not running or accessible. Please ensure your Docker container is up at http://localhost:8000.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {e}. Response: {e.response.text if e.response else 'No response text'}")

    st.write("---")
    st.write("Raw API Response (for debugging):")
    st.json(detections) # Always show raw JSON for debugging purposes