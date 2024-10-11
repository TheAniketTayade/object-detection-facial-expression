from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import requests
import cv2
import os

expression_thresh = 0.60
save_output_images = True

# Initialize YOLO model
model = YOLO("face_yolov8n.pt")
names = model.names

# Directory containing images
image_directory = "images"
output_directory = "output_images"

if save_output_images:
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

# Define the API endpoint
facial_expressions_api_url = "http://127.0.0.1:1981/predict"

# Loop through all the files in the directory
for image_name in os.listdir(image_directory):
    # Make sure we are only processing image files
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        # Read the image
        image_path = os.path.join(image_directory, image_name)
        im0 = cv2.imread(image_path)
        if im0 is None:
            print(f"Error reading image {image_name}")
            continue

        # Get the image dimensions
        h, w = im0.shape[:2]

        # Process the image with YOLO model
        results = model.predict(im0, show=False, conf=0.4, classes=0)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        annotator = Annotator(im0, line_width=2, example=names)

        # If boxes are detected
        if boxes is not None:
            for box, cls in zip(boxes, clss):

                # Extract object
                obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

                # Convert blur_obj to bytes for sending in API request
                _, img_encoded = cv2.imencode('.jpg', obj)
                files = {'image': ('blur_obj.jpg', img_encoded.tobytes(), 'image/jpeg')}

                # Send POST request to API
                response_expressions = requests.post(facial_expressions_api_url, files=files)

                # Parse the response
                if response_expressions.status_code == 200:
                    data_expressions = response_expressions.json()
                    print(f"Predicted Label: {data_expressions['predicted_label']}")

                    if float(data_expressions['confidence']) > expression_thresh:
                        expressions_label = data_expressions['predicted_label']
                    else:
                        expressions_label = 'neutral'

                    annotator.box_label(box, color=colors(int(cls), True), label=expressions_label)

        # Save the annotated image to the output directory
        if save_output_images:
            output_path = os.path.join(output_directory, image_name)
            cv2.imwrite(output_path, im0)
            print(f"Annotated image saved at {output_path}")
