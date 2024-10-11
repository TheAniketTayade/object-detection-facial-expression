
# Object Detection and Facial Expression Recognition Pipeline

This project integrates object detection using YOLOv8 with facial expression recognition using a Flask-based API. The detected objects in images are processed to recognize facial expressions, which are then annotated and saved in the output directory.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Facial Expression API](#facial-expression-api)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This pipeline performs two primary tasks:
1. **Object Detection using YOLOv8:** Detects objects (people) in the images.
2. **Facial Expression Recognition:** After detecting a person, the upper body portion of the bounding box is extracted and sent to a Flask API for facial expression recognition. The predicted expression is then used to annotate the detected objects in the image.
3. **Or Face Detection using YOLOv8:** Detects objects (Face) in the images and later process will be same.

## Requirements

To run this project, the following dependencies are required:

- Python 3.8+
- Required Python packages (see [Installation](#installation))

Ensure that you also have a working **Flask API** for facial expression recognition. The instructions to set it up are included in the [Facial Expression API](#facial-expression-api) section.

## Installation

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/TheAniketTayade/object-detection-facial-expression.git
cd object-detection-facial-expression
```

2. Install the required Python packages:

```bash
pip install ultralytics opencv-python-headless requests torch torchvision safetensors transformers
```

Ensure the facial expression recognition model is downloaded and placed in the same directory as the API code.
3. Go to vit-Facial-Expression-Recognition and download the model

```wget
wget "https://huggingface.co/motheecreator/vit-Facial-Expression-Recognition/resolve/main/model.safetensors?download=true" -O model.safetensors
```

## Usage

1. **Start the Facial Expression Recognition API**:
   The API must be running on `http://127.0.0.1:1981/predict` before executing the main script.

   Navigate to the API folder and run it:

   ```bash
   python app.py
   ```

2. **Run the Object Detection and Expression Recognition Script**:

   Once the API is running, you can execute the script:
   3. There are following scripts with different usecases
      4. api_test.py : just a basic script to show how to consume the API
      5. Yolo_expressions_detection_upperbody_images.py : This is detection code for dir of images using yoloV8n for detection and cropping out upper part of person and passing it to face expression API
      6. Yolo_expressions_detection_upperbody_video.py : Same as above but for video, it will process your video and save it.
      7. Yolo_expressions_detection_face_images.py : Using face detector yolov8n model and passing those faces to API
      8. Yolo_expressions_detection_face_video.py: same but for video
         ```bash
         python {your_script_name}.py
         ```

         The script will:
          - Detect objects (persons) using YOLOv8.
          - Extract the upper body from the detected persons.
          - Send the cropped image of the upper body to the API for expression recognition.
          - Annotate the image with the predicted facial expression.
          - Save the annotated images in the output directory.

## Directory Structure

```bash
├── images/             # Directory for input images
├── output_images/      # Directory where annotated images will be saved
├── api_test.py         # just a basic script to show how to consume the API
└── Yolo_expressions_detection_upperbody_images.py           # using person detector and extracting upper body for images
├── Yolo_expressions_detection_upperbody_video.py      # using person detector and extracting upper body for video
├── Yolo_gender_detection_face_images.py       # using face detector for images
├── Yolo_expressions_detection_face_video.py        # using face detector for video
```

- **images/**: This folder contains the input images for object detection and expression recognition.
- **output_images/**: Annotated images with bounding boxes and facial expressions are saved here.

## Configuration

The script comes with the following configuration options:

- **YOLO Model**: Uses `yolov8n.pt` for object detection. You can change this to another model if desired.
- **Expression Confidence Threshold**:
    - Set to `0.60` by default. Adjust this threshold (`expression_thresh`) to determine the minimum confidence for facial expression predictions.
    - If the prediction confidence is lower than the threshold, it labels the expression as 'neutral'.

- **Save Output Images**:
    - Set `save_output_images = True` to save the images annotated with bounding boxes and predicted expressions. The annotated images will be saved to the `output_images` directory.

## Facial Expression API

The script communicates with a Flask-based API to predict facial expressions. Ensure this API is running on `http://127.0.0.1:1981/predict`.

The API expects the following:
- **Method**: `POST`
- **Endpoint**: `/predict`
- **Input**: A JPEG image (upper body or face portion).
- **Output**: A JSON response with:
    - `predicted_label`: The predicted facial expression (e.g., 'happy', 'sad').
    - `confidence`: The confidence score of the prediction.

### API Example Response:

```json
{
  "predicted_label": "happy",
  "confidence": "0.85",
  "processing_time": "0.334 seconds"
}
```

## Output

Annotated images with detected bounding boxes and facial expressions will be saved in the `output_images` directory. Each detected object (person) will have the corresponding facial expression label annotated on the image.

## Contributing

Contributions are welcome! Please submit pull requests or open an issue to discuss changes or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
