# frontend/app.py
import streamlit as st
from PIL import Image
import io
import time

# Import our modularized functions
from api_client import check_api_health, predict, get_lambda_status_display, warm_up_lambda
from drawing import draw_predictions_on_image

# --- Page Configuration ---
st.set_page_config(
    page_title="YOLOv8 Vegetable Detection",
    page_icon="🥕",
    layout="wide"
)

# --- State Management ---
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'visibility_toggles' not in st.session_state:
    st.session_state.visibility_toggles = {}
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'api_status' not in st.session_state:
    st.session_state.api_status = "unknown"
if 'last_health_check' not in st.session_state:
    st.session_state.last_health_check = 0

# --- Header Section ---
st.title("🥕 YOLOv8 Object Detection Demo")
st.markdown("""
**Custom-trained YOLOv8 model deployed on AWS Lambda for serverless inference**

Detects and counts vegetables in uploaded images. Demonstrates end-to-end ML pipeline from training to production deployment.
""")

# --- Technical Overview ---
with st.expander("🔧 Technical Stack", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model & Training:**
        - YOLOv8 (Ultralytics)
        - Custom dataset: Carrots, Beans, Radish
        - Transfer learning from COCO weights
        - Optimized for serverless deployment
        """)
    
    with col2:
        st.markdown("""
        **Deployment & Infrastructure:**
        - AWS Lambda (containerized)
        - API Gateway endpoint
        - Streamlit Cloud frontend
        - **Cost: $0/month** (free tier optimized)
        """)

    st.markdown("""
    **MLOps Considerations:**
    - Container size optimization for Lambda
    - Cold start handling (~30-60s first request)
    - Retry logic for serverless reliability
    - Real-time inference with cost efficiency
    
    **GitHub Repository:** https://github.com/fib-spiral/ml-inventory-counter/tree/main
    """)

# --- Enhanced Sidebar ---
def display_lambda_status():
    """Display enhanced Lambda status with better messaging"""
    status_type, message, color = get_lambda_status_display()
    
    if status_type == "success":
        st.success(message)
    elif status_type == "warning":
        st.warning(message)
        st.caption("First request will initialize the function")
    elif status_type == "error":
        st.error(message)
        st.caption("Lambda may need deployment or warm-up")
    else:
        st.info(message)

with st.sidebar:
    st.header("🔧 System Status")
    
    # Enhanced API Status display
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        display_lambda_status()
    
    with col2:
        if st.button("🔄", help="Refresh Status"):
            st.session_state.last_health_check = 0
            st.rerun()
    
    with col3:
        if st.button("🚀", help="Warm Up Lambda"):
            with st.spinner("Warming up..."):
                if warm_up_lambda():
                    st.success("Warmed!")
                else:
                    st.error("Failed")
    
    st.markdown("---")
    
    # Model Information
    st.header("📊 Model Details")
    st.markdown("""
    **Trained Classes:**
    - 🥕 Carrots
    - 🫘 Beans  
    - 🌶️ Radish
    
    **Upload images containing these items only.**
    """)
    
    st.markdown("---")

    # Display toggles ONLY if there are predictions to show
    if st.session_state.predictions and st.session_state.predictions.get("predictions"):
        st.header("🎛️ Display Options")
        
        # Create a master toggle
        all_visible = all(st.session_state.visibility_toggles.values())
        if st.checkbox("Toggle All Bounding Boxes", value=all_visible):
            for cls in st.session_state.visibility_toggles:
                st.session_state.visibility_toggles[cls] = True
        else:
            if all_visible:
                for cls in st.session_state.visibility_toggles:
                    st.session_state.visibility_toggles[cls] = False

        st.subheader("Detected Classes")
        for class_name in sorted(st.session_state.visibility_toggles.keys()):
            st.session_state.visibility_toggles[class_name] = st.checkbox(
                class_name.capitalize(), 
                value=st.session_state.visibility_toggles[class_name],
                key=f"toggle_{class_name}"
            )

# --- Main Application ---
st.markdown("---")
st.header("🖼️ Image Upload & Detection")

# Important usage note
st.info("""
**📝 Usage Instructions:**
1. Upload an image containing **carrots, beans, or radish**
2. Click "Detect Vegetables" to run inference
3. **First request may take 30-60 seconds** due to Lambda cold start
4. **Subsequent requests will be much faster** (~2-5 seconds)
5. **Try the warm-up button** in the sidebar to pre-initialize Lambda

**💡 This is a proof of concept** - the same technology scales to any inventory category!
""")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image (JPG, JPEG, PNG)...", 
    type=["jpg", "jpeg", "png"],
    key="file_uploader"
)

if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    st.session_state.original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Reset predictions when a new image is uploaded
    if st.session_state.file_uploader and uploaded_file.name != st.session_state.get('last_uploaded_name'):
        st.session_state.predictions = None
        st.session_state.visibility_toggles = {}
    st.session_state.last_uploaded_name = uploaded_file.name

    # Enhanced detection button with better feedback
    if st.button("🔍 Detect Vegetables", key="detect_button", type="primary"):
        # Check current Lambda status
        status_type, status_message, _ = get_lambda_status_display()
        
        if status_type == "warning":
            st.warning("⚠️ Lambda is cold. This may take 30-60 seconds to initialize...")
        elif status_type == "error":
            st.error("⚠️ Lambda appears unreachable. Attempting connection...")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Connecting to Lambda endpoint...")
            progress_bar.progress(20)
            
            status_text.text("Uploading image for analysis...")
            progress_bar.progress(40)
            
            # Use the enhanced predict function with retry logic
            detections = predict(image_bytes, uploaded_file.name)
            progress_bar.progress(80)
            
            status_text.text("Processing detection results...")
            progress_bar.progress(95)
            
            if detections and detections.get("predictions"):
                st.session_state.predictions = detections
                all_classes = {det["class_name"] for det in detections["predictions"]}
                st.session_state.visibility_toggles = {cls: True for cls in all_classes}
                status_text.text("✅ Detection completed successfully!")
                progress_bar.progress(100)
                time.sleep(0.5)
                st.success("🎉 Detection completed! Results shown below.")
                
            elif detections:  # Valid response but no predictions
                st.session_state.predictions = detections
                st.session_state.visibility_toggles = {}
                status_text.text("No vegetables detected")
                progress_bar.progress(100)
                time.sleep(0.5)
                st.info("📋 No vegetables detected. Try an image with carrots, beans, or radish.")
                
            else:  # API call failed
                st.session_state.predictions = None
                st.session_state.visibility_toggles = {}
                status_text.text("❌ Detection failed")
                progress_bar.progress(0)
                # Error handling is now done in the enhanced predict function
                
        except Exception as e:
            st.session_state.predictions = None
            st.session_state.visibility_toggles = {}
            status_text.text("❌ Unexpected error")
            progress_bar.progress(0)
            st.error(f"Unexpected error: {str(e)}")
        
        finally:
            # Clean up progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

# --- Results Display ---
if st.session_state.predictions:
    st.markdown("---")
    st.header("📊 Detection Results")
    
    # Draw the image based on current toggle states
    annotated_image, counts = draw_predictions_on_image(
        st.session_state.original_image, 
        st.session_state.predictions,
        st.session_state.visibility_toggles
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.original_image, use_column_width=True)
    with col2:
        st.subheader("Detected Items")
        st.image(annotated_image, use_column_width=True)
        
    # Enhanced counts display
    st.subheader("🔢 Item Counts")
    if counts:
        # Create a more visually appealing counts display
        count_cols = st.columns(len(counts))
        for i, (item, count) in enumerate(sorted(counts.items())):
            with count_cols[i]:
                # Add emojis for each vegetable type
                emoji_map = {"carrot": "🥕", "bean": "🫘", "radish": "🌶️"}
                emoji = emoji_map.get(item.lower(), "🥬")
                st.metric(
                    label=f"{emoji} {item.capitalize()}",
                    value=count,
                    help=f"Number of {item}s detected"
                )
    else:
        st.info("No items detected in this image, or all bounding boxes are hidden.")

    # Technical details in expandable section
    with st.expander("🔧 Technical Details"):
        st.markdown("**Detection Results:**")
        if st.session_state.predictions.get("predictions"):
            for i, pred in enumerate(st.session_state.predictions["predictions"]):
                st.write(f"**{i+1}.** {pred['class_name']} (confidence: {pred['confidence']:.2%})")
        
        st.markdown("**Raw API Response:**")
        st.json(st.session_state.predictions)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>
        <strong>YOLOv8 • AWS Lambda • Streamlit Cloud</strong>
        <br>
        <em>Demonstrating ML model deployment and serverless inference</em>
    </p>
</div>
""", unsafe_allow_html=True)