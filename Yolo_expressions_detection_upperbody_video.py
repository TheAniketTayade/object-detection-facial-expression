from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import requests
import cv2
import numpy as np
import io

expression_thresh = 0.60

model = YOLO("yolov8n.pt")
names = model.names

cap = cv2.VideoCapture("test.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, (w, h))

# Define the API endpoint
facial_expressions_api_url = "http://127.0.0.1:1981/predict"

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, show=False, conf=0.4, classes=0)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=names)

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            # Extract object
            obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            # Calculate the centroid of the bounding box
            centroid_x = (box[0] + box[2]) / 2
            centroid_y = (box[1] + box[3]) / 2

            # Calculate the middle y-coordinate
            middle_y = int(box[1]) + (int(box[3]) - int(box[1])) // 2

            # Slice the upper half of the bounding box
            upper_body = im0[int(box[1]):middle_y, int(box[0]):int(box[2])]

            # Convert upper_body to bytes for sending in API request
            _, img_encoded = cv2.imencode('.jpg', upper_body)
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

    # Write the frame to output video
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
