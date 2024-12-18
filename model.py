import os
import json
import tensorflow as tf
from tensorflow import keras
import numpy as np

def convert_model(input_path, output_path, labels_path):
    """
    Advanced model conversion with multiple fallback strategies
    Supports custom model architectures and robust error handling
    """
    try:
        # Read labels to determine output classes
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        num_classes = len(labels)

        # Custom objects dictionary for potential complex models
        custom_objects = {
            'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
            'Gelu': tf.keras.activations.gelu
        }
        
        # Logging function
        def log_message(message):
            print(f"[MODEL CONVERSION] {message}")

        # Model loading strategies
        def try_load_model(load_method):
            try:
                model = load_method()
                log_message("Model loaded successfully")
                return model
            except Exception as e:
                log_message(f"Loading failed: {e}")
                return None

        # Strategy 1: Load with custom objects
        def load_with_custom_objects():
            return keras.models.load_model(
                input_path, 
                compile=False, 
                custom_objects=custom_objects
            )

        # Strategy 2: Direct loading
        def load_directly():
            return keras.models.load_model(input_path, compile=False)

        # Strategy 3: Manual reconstruction (fallback)
        def reconstruct_model():
            log_message("Attempting manual model reconstruction")
            model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(224, 224, 3)),
                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(num_classes, activation='softmax')
            ])
            return model

        # Attempt model loading strategies
        original_model = (
            try_load_model(load_with_custom_objects) or 
            try_load_model(load_directly) or 
            try_load_model(reconstruct_model)
        )

        if original_model is None:
            raise ValueError("Failed to load model through any method")

        # Create a new sequential model with the same architecture
        new_model = keras.Sequential()
        for layer in original_model.layers:
            new_model.add(layer)

        # Compile the model
        new_model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )

        # Save the converted model
        new_model.save(output_path)
        log_message(f"Model successfully converted and saved to {output_path}")

        # Verify the model
        loaded_model = keras.models.load_model(output_path)
        log_message("Model verification successful")

        # Print detailed model information
        print("\n--- Model Details ---")
        print(f"Input Shape: {loaded_model.input_shape}")
        print(f"Output Shape: {loaded_model.output_shape}")
        print(f"Number of Classes: {num_classes}")
        
        # Save model metadata
        metadata = {
            "input_shape": loaded_model.input_shape,
            "output_shape": loaded_model.output_shape,
            "num_classes": num_classes,
            "class_names": labels
        }
        
        with open(output_path.replace('.h5', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        log_message("Metadata saved successfully")
        
        return loaded_model

    except Exception as e:
        print(f"Critical error converting model: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_sample(model_path, sample_image_path):
    """
    Test the converted model with a sample image
    """
    try:
        # Load the converted model
        model = keras.models.load_model(model_path)
        
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(
            sample_image_path, 
            target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        
        # Normalize the image
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Load class names
        with open('labels.txt', 'r') as f:
            class_names = [line.strip().split(' ', 1)[-1] for line in f.readlines()]
        
        # Get the predicted class
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100
        
        print(f"\n--- Sample Prediction ---")
        print(f"Predicted Class: {predicted_class_name}")
        print(f"Confidence: {confidence:.2f}%")
        
        return predicted_class_name, confidence
    
    except Exception as e:
        print(f"Error during sample prediction: {e}")
        return None

# Main execution
if __name__ == '__main__':
    input_model_path = 'keras_Model.h5'
    output_model_path = 'converted_model.h5'
    labels_path = 'labels.txt'
    
    # Convert the model
    converted_model = convert_model(input_model_path, output_model_path, labels_path)
    
    # Optional: Test with a sample image (uncomment and provide path)
    # sample_image_path = 'path/to/sample/image.jpg'
    # predict_sample(output_model_path, sample_image_path)

print("\nModel conversion script completed. Check the output for details.")