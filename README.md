# Environment-Engineering

This project demonstrates how to use YOLOv5 for object detection and compares the performance of inference on original versus sharpened images.

## Table of Contents
- [Setup](#setup)
- [Features](#features)
- [Usage](#usage)
  - [1. Setting up the environment](#11-setting-up-the-environment)
  - [2. Loading YOLO Neural Network](#12-loading-yolo-neural-network)
  - [3. Auxiliary Functions](#13-auxiliary-functions)
  - [4. Adding Testing Images](#14-adding-testing-images)
  - [5. Running Main Code](#15-running-main-code)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Setup

Follow the provided Jupyter Notebook to replicate the code and workflow. 

### Prerequisites
- Python (>=3.8 recommended)
- PyTorch
- PIL (Python Imaging Library)

Clone the YOLOv5 repository:

```bash
git clone https://github.com/ultralytics/yolov5.git


import torch
from PIL import Image, ImageFilter
!git clone https://github.com/ultralytics/yolov5.git


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # For larger models, use 'yolov5m', 'yolov5l', or 'yolov5x'


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

!wget -O /content/temp_image1.jpg 'https://media.npr.org/assets/img/2024/01/09/gettyimages-1258833682-bdd8ee9eddc072e4ffe6590b7e7c3a58cfe4f54a.jpg'
!wget -O /content/temp_image2.jpg 'https://www.shutterstock.com/image-photo/plastic-bottle-on-street-blurry-260nw-1124703005.jpg'
!wget -O /content/temp_image3.jpg 'https://www.shutterstock.com/shutterstock/photos/2112828356/display_1500/stock-photo-blurred-counter-has-a-wide-variety-of-liquor-bottles-that-are-sold-in-supermarkets-blur-bottles-of-2112828356.jpg'
!wget -O /content/temp_image4.jpg 'https://images.stockcake.com/public/9/b/0/9b0615bc-e810-4a0d-a372-0ef2cd5a4f2a_large/filling-water-bottle-stockcake.jpg'

# List of image paths
image_paths = ['/content/temp_image1.jpg', '/content/temp_image2.jpg', '/content/temp_image3.jpg', '/content/temp_image4.jpg']
