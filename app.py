import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image, ImageOps
import io

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load labels
def load_labels(file_path):
    """Load class labels from a text file"""
    try:
        with open(file_path, 'r') as f:
            # Strip whitespace and remove potential numbering
            return [line.strip().split(' ', 1)[-1] for line in f.readlines()]
    except Exception as e:
        print(f"Error loading labels: {e}")
        return []

# Advanced model loading with multiple strategies
def load_keras_model(model_path):
    """
    Load Keras model with robust error handling and multiple strategies
    """
    def custom_depthwise_conv2d(**kwargs):
        # Custom handling for DepthwiseConv2D with problematic configuration
        kwargs.pop('groups', None)  # Remove 'groups' if present
        return tf.keras.layers.DepthwiseConv2D(**kwargs)

    custom_objects = {
        'DepthwiseConv2D': custom_depthwise_conv2d,
        'tf': tf
    }

    try:
        # Strategy 1: Load with custom objects and error handling
        model = tf.keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects=custom_objects
        )
        return model
    
    except Exception as e:
        print(f"Primary model loading failed: {e}")
        
        try:
            # Strategy 2: More aggressive custom object handling
            with tf.keras.utils.custom_object_scope(custom_objects):
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False
                )
            return model
        
        except Exception as e:
            print(f"Secondary loading strategy failed: {e}")
            
            try:
                # Strategy 3: Ignore custom layers and rebuild
                base_model = tf.keras.models.load_model(
                    model_path, 
                    compile=False,
                    custom_objects={'DepthwiseConv2D': tf.keras.layers.Conv2D}
                )
                return base_model
            
            except Exception as e:
                print(f"Fallback model loading failed: {e}")
                return None

# Load model and labels
try:
    model = load_keras_model("keras_Model.h5")
    class_names = load_labels("labels.txt")
    
    if model is None:
        raise ValueError("Model could not be loaded")
except Exception as e:
    print(f"Critical error: {e}")
    model = None
    class_names = []

def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image):
    """Prepare image for model prediction"""
    # Resize the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Reshape for model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    return data

def predict_freshness(image_data):
    """Predict freshness using the Keras model"""
    if model is None:
        return {'error': 'Model not loaded'}
    
    # Predict the model
    prediction = model.predict(image_data)
    index = np.argmax(prediction)
    
    # Ensure we have valid class names
    if 0 <= index < len(class_names):
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        return {
            'class': class_name,
            'confidence': float(np.round(confidence_score * 100, 2))
        }
    else:
        return {'error': 'Invalid prediction index'}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Main route for file upload and prediction"""
    if request.method == 'POST':
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Machine learning model is not available'}), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Special handling for camera capture (might be a bytes file)
        if hasattr(file, 'content_type'):
            file_extension = file.content_type.split('/')[-1]
            filename = f'capture.{file_extension}'
        else:
            filename = secure_filename(file.filename)
        
        # Check if file is allowed
        if file:
            try:
                # Open image directly from file storage
                image = Image.open(io.BytesIO(file.read())).convert("RGB")
                prepared_image = prepare_image(image)
                
                # Predict freshness
                result = predict_freshness(prepared_image)
                
                return jsonify(result)
            
            except Exception as e:
                return jsonify({'error': f'Image processing failed: {str(e)}'}), 500
    
    # Render the upload page for GET requests
    return render_template('index.html')

@app.route('/diagnostics')
def diagnostics():
    """Provide system diagnostics"""
    return jsonify({
        'model_loaded': model is not None,
        'num_classes': len(class_names),
        'classes': class_names
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')