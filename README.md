This project demonstrates how to use YOLOv5 for object detection and compares the performance of inference on original versus sharpened images.

Table of Contents
Setup
Features
Usage
1. Setting up the environment
2. Loading YOLO Neural Network
3. Auxiliary Functions
4. Adding Testing Images
5. Running Main Code
Results
Acknowledgments
Setup
Follow the provided Jupyter Notebook to replicate the code and workflow.

Prerequisites
Python (>=3.8 recommended)
PyTorch
PIL (Python Imaging Library)
Clone the repository:

bash
Copy code
git clone https://github.com/ultralytics/yolov5.git
Features
Perform object detection using YOLOv5.
Apply unsharp masking for image sharpening.
Compare detection results on original and sharpened images.
Usage
1.1 Setting up the environment
Install the required libraries and clone the YOLOv5 repository:

python
Copy code
import torch
from PIL import Image, ImageFilter
!git clone https://github.com/ultralytics/yolov5.git
1.2 Loading YOLO Neural Network
Load the YOLOv5 model:

python
Copy code
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # For larger models, use 'yolov5m', 'yolov5l', or 'yolov5x'
1.3 Auxiliary Functions
Define helper functions for image sharpening and running inference:

python
Copy code
def sharpen_image(image_path):
    """
    Apply unsharp masking to an image.
    """
    img = Image.open(image_path)
    return img.filter(ImageFilter.UnsharpMask(radius=8, percent=160, threshold=5))

def run_inference(image):
    """
    Perform inference on an image using YOLOv5.
    """
    results = model(image)
    return results
1.4 Adding Testing Images
Download test images:

bash
Copy code
!wget -O /content/temp_image1.jpg 'https://media.npr.org/assets/img/2024/01/09/gettyimages-1258833682-bdd8ee9eddc072e4ffe6590b7e7c3a58cfe4f54a.jpg'
!wget -O /content/temp_image2.jpg 'https://www.shutterstock.com/image-photo/plastic-bottle-on-street-blurry-260nw-1124703005.jpg'
!wget -O /content/temp_image3.jpg 'https://www.shutterstock.com/shutterstock/photos/2112828356/display_1500/stock-photo-blurred-counter-has-a-wide-variety-of-liquor-bottles-that-are-sold-in-supermarkets-blur-bottles-of-2112828356.jpg'
!wget -O /content/temp_image4.jpg 'https://images.stockcake.com/public/9/b/0/9b0615bc-e810-4a0d-a372-0ef2cd5a4f2a_large/filling-water-bottle-stockcake.jpg'

# List of image paths
image_paths = ['/content/temp_image1.jpg', '/content/temp_image2.jpg', '/content/temp_image3.jpg', '/content/temp_image4.jpg']
1.5 Running Main Code
Process each image, display results for original and sharpened images:

python
Copy code
for img_path in image_paths:
    print(f"Processing image: {img_path}")

    # Run inference on the original image
    results = run_inference(img_path)
    print("Inference on original image:")
    results.show()

    # Sharpen the image
    sharpened_img = sharpen_image(img_path)

    # Save and run inference on the sharpened image
    sharpened_img_path = f'/{img_path[1:-4]}_sharp.jpg'
    sharpened_img.save(sharpened_img_path)

    results = run_inference(sharpened_img_path)
    print("Inference on sharpened image:")
    results.show()
Results
Observations
Better results with original images: Some images perform better without sharpening.
Similar results: Original and sharpened images yield similar results for certain cases.
Improved detection with sharpening: Sharpened images detect additional objects.
Detection on sharpened images only: Objects are detected exclusively in sharpened images in specific scenarios.
Acknowledgments
YOLOv5 GitHub Repository
Test images are sourced from publicly available repositories and licensed appropriately.
