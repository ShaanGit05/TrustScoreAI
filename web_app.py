import sys
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠ Warning: Could not load .env file: {e}")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, request, jsonify, send_from_directory
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

from src.pipeline import BiasDetectionPipeline
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bias_detection.web")

# Create Flask app
app = Flask(__name__, 
           static_folder='frontend',
           static_url_path='')
app.config['JSON_SORT_KEYS'] = False

if CORS_AVAILABLE:
    CORS(app)

# Global pipeline instance
pipeline = None

def initialize_pipeline():
    """Initialize the bias detection pipeline."""
    global pipeline
    try:
        pipeline = BiasDetectionPipeline()
        logger.info("Pipeline initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return False

# Initialize pipeline on import
initialize_pipeline()

@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('frontend', filename)

@app.route('/api/models')
def get_models():
    """Get list of supported models."""
    logger.info("Models API endpoint called")
    
    if not pipeline:
        logger.error("Pipeline not initialized")
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    try:
        models = pipeline.get_supported_models()
        logger.info(f"Retrieved {len(models)} models: {models}")
        return jsonify({'models': models})
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_model():
    """Analyze bias for a specific model."""
    if not pipeline:
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        logger.info(f"Received analysis request for model: {model_name}")
        
        # Validate model support
        if not pipeline.validate_model_support(model_name):
            return jsonify({'error': f'Model {model_name} is not supported'}), 400
        
        # Run bias analysis
        results = pipeline.run_pipeline(model_name)
        
        # Build metadata (shared)
        metadata = {
            'timestamp': results['metadata']['timestamp'],
            'processing_time': results['metadata']['processing_time_seconds'],
            'total_prompts': results['metadata']['total_prompts']
        }
        
        # Prepare response: backward-compatible top-level + new global/by_category structure
        response = {
            'model_name': model_name,
            # Backward compatibility: flat structure for existing consumers
            'ubi_score': results['ubi_score'],
            'bias_level': results['bias_level'],
            'components': results['components'],
            # New structure: explicit global + category-wise
            'global': {
                'ubi_score': results['ubi_score'],
                'bias_level': results['bias_level'],
                'components': {
                    'BM': results['components']['bias_magnitude'],
                    'DP': results['components']['disparity'],
                    'DS': results['components']['distribution_shift']
                }
            },
            'by_category': results.get('by_category', {}),
            'weights': results['weights'],
            'metadata': metadata
        }
        
        logger.info(f"Analysis completed for {model_name}: UBI={results['ubi_score']:.4f}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Get API status."""
    pipeline_status = "initialized" if pipeline else "not initialized"
    return jsonify({
        'status': 'running',
        'pipeline': pipeline_status,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Bias Detection Web Interface...")
    print("📍 Web interface will be available at: http://localhost:5000")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Initialize pipeline
    if initialize_pipeline():
        app.run(debug=True, host='0.0.0.0', port=8000)
    else:
        logger.error("Failed to start application: Pipeline initialization failed")
