import time
from flask import Flask, request, jsonify
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from torch.nn import functional as F
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load image processor and model
image_processor = ViTImageProcessor.from_pretrained("./vit-Facial-Expression-Recognition")
model = ViTForImageClassification.from_pretrained("./vit-Facial-Expression-Recognition", use_safetensors=True)

# Function to run predictions on an image
def predict(image):
    # Preprocess the image
    inputs = image_processor(images=image, return_tensors="pt")

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Get predicted class and confidence
        probabilities = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
        predicted_class_idx = logits.argmax(-1).item()
        confidence = probabilities[0][predicted_class_idx].item()  # Confidence for predicted class

    # Get label (adjust according to model configuration)
    label = model.config.id2label[predicted_class_idx]
    return label, confidence

# API endpoint to handle image uploads
@app.route('/predict', methods=['POST'])
def predict_api():
    # Get the image from the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")  # Convert the image to RGB

        # Start timing
        start_time = time.monotonic()

        # Run prediction
        prediction, confidence = predict(image)

        # End timing
        end_time = time.monotonic() - start_time

        # Return the result as JSON
        return jsonify({
            'predicted_label': prediction,
            'confidence': f"{confidence:.4f}",  # Return confidence with 4 decimal precision
            'processing_time': f"{end_time:.4f} seconds"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False, port=1981)
