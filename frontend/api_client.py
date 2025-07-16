# frontend/api_client.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv
import time
from typing import Optional, Dict, Any

# --- Configuration ---

def get_api_base_url():
    """
    Determines the API base URL using a robust hierarchy that is safe for local execution.
    It uses an environment variable (IS_CLOUD_DEPLOYMENT) to switch between cloud and local logic.
    """
    # --- CHECK 1: Are we running in the cloud? ---
    if os.environ.get("IS_CLOUD_DEPLOYMENT") == "true":
        st.sidebar.info("🎯 Using Streamlit Cloud Secrets (Live Mode)")
        return st.secrets.get("API_GATEWAY_URL", "") 

    load_dotenv()
    
    # CHECK 2: Is there a local .env file with the URL?
    local_env_url = os.environ.get("API_GATEWAY_URL")
    if local_env_url:
        st.sidebar.info("🎯 Using local .env URL (Local Live Mode)")
        return local_env_url
    
    # CHECK 3: Fallback to Docker Compose service name.
    st.sidebar.info("🎯 Using local Docker service (Local Container Mode)")
    api_service_name = "vegetable-counter-api"  
    port = 8000
    return f"http://{api_service_name}:{port}"

# --- Enhanced API Functions ---
API_BASE_URL = get_api_base_url()

def check_api_health_detailed() -> Dict[str, Any]:
    """
    Enhanced health check that provides more detailed status information.
    Returns a dictionary with status, timing, and error details.
    """
    start_time = time.time()
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            return {
                "status": "healthy",
                "status_code": "200",
                "response_time": response_time,
                "message": "Lambda function is warm and ready",
                "data": response.json()
            }
        else:
            return {
                "status": "unhealthy",
                "status_code": str(response.status_code),
                "response_time": response_time,
                "message": f"API returned status {response.status_code}",
                "error": response.text
            }
            
    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "status_code": "timeout",
            "response_time": time.time() - start_time,
            "message": "Lambda function may be cold - initialization needed",
            "error": "Request timed out during health check"
        }
    except requests.exceptions.ConnectionError:
        return {
            "status": "unreachable",
            "status_code": "connection_error",
            "response_time": time.time() - start_time,
            "message": "Cannot connect to Lambda endpoint",
            "error": "Connection failed - check if Lambda is deployed"
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "status_code": "request_error",
            "response_time": time.time() - start_time,
            "message": "Unexpected error during health check",
            "error": str(e)
        }

@st.cache_data(ttl=30)  # Reduced TTL for more responsive status updates
def check_api_health():
    """
    Cached version of health check for backward compatibility.
    Uses the detailed health check internally.
    """
    detailed_status = check_api_health_detailed()
    return {
        "status_code": detailed_status["status_code"],
        "status": detailed_status["status"]
    }

def predict_with_retry(image_bytes: bytes, image_name: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
    """
    Enhanced prediction function with retry logic and better error handling.
    
    Args:
        image_bytes: The image data as bytes
        image_name: Name of the uploaded image
        max_retries: Maximum number of retry attempts for cold start scenarios
    
    Returns:
        Dictionary containing prediction results or None if failed
    """
    predict_endpoint = f"{API_BASE_URL}/predict"
    files = {"file": (image_name, image_bytes, "image/jpeg")}
    
    for attempt in range(max_retries + 1):
        try:
            # Progressively increase timeout for retries (cold start consideration)
            timeout = 60 + (attempt * 30)  # 60s, 90s, 120s
            
            if attempt > 0:
                st.info(f"🔄 Retry attempt {attempt} - Lambda may be initializing...")
                time.sleep(2)  # Brief delay between retries
            
            response = requests.post(predict_endpoint, files=files, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate response structure
                if "predictions" in result:
                    return result
                else:
                    st.warning("⚠️ API returned unexpected response format")
                    return None
                    
            elif response.status_code == 502:
                if attempt < max_retries:
                    st.warning(f"🔄 Lambda cold start detected (502 error). Retrying... (Attempt {attempt + 1}/{max_retries + 1})")
                    continue
                else:
                    st.error("❌ Lambda failed to initialize after multiple attempts")
                    return None
                    
            elif response.status_code == 504:
                if attempt < max_retries:
                    st.warning(f"⏰ Lambda timeout (504 error). Retrying with longer timeout... (Attempt {attempt + 1}/{max_retries + 1})")
                    continue
                else:
                    st.error("❌ Lambda consistently timing out - may need longer initialization time")
                    return None
                    
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                st.warning(f"⏰ Request timeout on attempt {attempt + 1}. This is normal for cold Lambda starts. Retrying...")
                continue
            else:
                st.error("❌ Request consistently timing out. Lambda may need manual warm-up.")
                return None
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to Lambda endpoint. Check deployment status.")
            return None
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None
    
    return None

# Backward compatibility
def predict(image_bytes: bytes, image_name: str) -> Optional[Dict[str, Any]]:
    """
    Backward compatible predict function that uses the enhanced version.
    """
    return predict_with_retry(image_bytes, image_name)

def warm_up_lambda() -> bool:
    """
    Attempts to warm up the Lambda function by making a lightweight request.
    Returns True if successful, False otherwise.
    """
    try:
        # Make a simple health check to warm up the function
        response = requests.get(f"{API_BASE_URL}/health", timeout=45)
        return response.status_code == 200
    except:
        return False

def get_lambda_status_display() -> tuple[str, str, str]:
    """
    Returns a tuple of (status_type, status_message, status_color) for UI display.
    
    Returns:
        tuple: (status_type, message, color) where:
            - status_type: "success", "warning", "error", or "info"
            - message: Human-readable status message
            - color: Color indicator for the status
    """
    health_status = check_api_health_detailed()
    
    if health_status["status"] == "healthy":
        return ("success", "✅ Lambda API: Ready & Warm", "green")
    elif health_status["status"] == "timeout":
        return ("warning", "⏰ Lambda API: Cold Start (Initializing)", "orange")
    elif health_status["status"] == "unreachable":
        return ("error", "❌ Lambda API: Unreachable", "red")
    else:
        return ("info", "🔄 Lambda API: Checking Status", "blue")