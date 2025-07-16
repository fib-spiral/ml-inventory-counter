# frontend/drawing.py
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

def draw_predictions_on_image(original_image, detections, visibility_toggles):
    """
    Draws bounding boxes and labels on an image based on detection results
    and visibility toggles.
    
    Returns:
        - The image with drawn bounding boxes.
        - A dictionary of detected item counts.
    """
    draw_image = original_image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    detected_counts = defaultdict(int)
    
    if detections and detections.get("predictions"):
        for det in detections["predictions"]:
            class_name = det["class_name"]
            detected_counts[class_name] += 1
            
            # Check the visibility toggle for this class
            if visibility_toggles.get(class_name, True): # Default to visible
                box = det["bounding_box"]
                confidence = det["confidence"]
                
                # Draw rectangle
                draw.rectangle([box["x1"], box["y1"], box["x2"], box["y2"]], 
                               outline="red", width=3)
                
                # Draw label
                label = f"{class_name} ({confidence:.2f})"
                text_x = box["x1"]
                text_y = box["y1"] - 25 if box["y1"] - 25 > 0 else box["y1"] + 5
                
                text_bbox = draw.textbbox((text_x, text_y), label, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((text_x, text_y), label, fill="white", font=font)
                
    return draw_image, detected_counts