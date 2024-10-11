import requests

# Define the API endpoint
api_url = "http://127.0.0.1:1981/predict"

# Path to the image file
image_path = "test1.png"

# Open the image file in binary mode and send a POST request
with open(image_path, 'rb') as image_file:
    files = {'image': image_file}
    response = requests.post(api_url, files=files)

# Handle the response
if response.status_code == 200:
    data = response.json()
    print(f"Predicted Label: {data['predicted_label']}")
    print(f"Processing Time: {data['processing_time']}")
    print(f"Processing Time: {data['confidence']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
