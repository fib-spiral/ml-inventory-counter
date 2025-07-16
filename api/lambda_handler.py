"""
Lambda handler that wraps the FastAPI app with Mangum
This file is only used when deploying to AWS Lambda
"""

from mangum import Mangum
from prediction_api import app
import logging

# Configure logging for Lambda
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the Lambda handler
handler = Mangum(app, lifespan="on")

# Optional: Add Lambda-specific initialization here
def lambda_handler(event, context):
    """
    AWS Lambda handler function
    
    Args:
        event: Lambda event object
        context: Lambda context object
        
    Returns:
        Response from FastAPI app via Mangum
    """
    
    # Log the incoming event (be careful with sensitive data)
    logger.info(f"Received Lambda event: {event.get('httpMethod', 'unknown')} {event.get('path', 'unknown')}")
    
    # Optional: Add any Lambda-specific preprocessing here
    # For example, you might want to modify headers or handle specific AWS services
    
    try:
        # Let Mangum handle the ASGI -> Lambda conversion
        response = handler(event, context)
        
        # Optional: Add any Lambda-specific postprocessing here
        logger.info("Request processed successfully")
        
        return response
        
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        # Return a proper Lambda response format
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": '{"success": false, "error": "Internal server error", "status_code": 500}'
        }

# For backwards compatibility, you can also export handler directly
# handler = lambda_handler