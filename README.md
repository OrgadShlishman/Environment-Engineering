# Environment-Engineering
Follow the Jupyiter Notebook to see the code

#### 1.1 Setting up environment
import torch
from PIL import Image, ImageFilter

!git clone https://github.com/ultralytics/yolov5.git

#### 1.2 Loading YOLO NN
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can use 'yolov5m', 'yolov5l', or 'yolov5x' for larger models

#### 1.3 Auxilary Functions
def sharpen_image(image_path):
    """
    Apply unsharp masking to an image.
    """
    img = Image.open(image_path)
    return img.filter(ImageFilter.UnsharpMask(radius=8, percent=160, threshold=5))

def run_inference(image):
    # Perform inference
    results = model(image)
    return results

#### 1.4 Adding testing images
!wget -O /content/temp_image1.jpg 'https://media.npr.org/assets/img/2024/01/09/gettyimages-1258833682-bdd8ee9eddc072e4ffe6590b7e7c3a58cfe4f54a.jpg'
!wget -O /content/temp_image2.jpg 'https://www.shutterstock.com/image-photo/plastic-bottle-on-street-blurry-260nw-1124703005.jpg'

!wget -O /content/temp_image3.jpg 'https://www.shutterstock.com/shutterstock/photos/2112828356/display_1500/stock-photo-blurred-counter-has-a-wide-variety-of-liquor-bottles-that-are-sold-in-supermarkets-blur-bottles-of-2112828356.jpg'
!wget -O /content/temp_image4.jpg 'https://images.stockcake.com/public/9/b/0/9b0615bc-e810-4a0d-a372-0ef2cd5a4f2a_large/filling-water-bottle-stockcake.jpg'

# List of image paths
image_paths = ['/content/temp_image1.jpg', '/content/temp_image2.jpg', '/content/temp_image3.jpg', '/content/temp_image4.jpg']


#### 1.5 Running main code and displaying results
In the following results one can see:

(1) Original Image shows better results than sharpened one

(2) Original and sharpened results are the same

(3) Original image detect few bottles, sharpened image detect more bottles

(4) Original image doesn't detect the bottle, sharpened image detect it
for img_path in image_paths:
    print(f"Processing image: {img_path}")

    # Display original image
    # original_img = Image.open(img_path)
    # display(original_img)

    # Run inference on the original image
    results = run_inference(img_path)
    print("Inference on original image:")
    results.show()

    # Sharpen the image
    sharpened_img = sharpen_image(img_path)

    # Save the sharpened image to a temporary path
    sharpened_img_path = f'/{img_path[1:-4]}_sharp.jpg'
    sharpened_img.save(sharpened_img_path)

    # Display sharpened image
    # print(f"Processing sharpened image: {sharpened_img_path}")
    # display(sharpened_img)

    # Run inference on the sharpened image
    results = run_inference(sharpened_img_path)
    print("Inference on sharpened image:")
    results.show()
