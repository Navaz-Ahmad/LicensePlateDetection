import cv2
import numpy as np
from torchvision import transforms

def preprocess_image(image, target_size=(640, 640)):
    """
    Preprocess an image for YOLO model input.
    - Resize the image to target size (640x640).
    - Normalize the image using typical values for YOLOv7.

    Args:
        image (ndarray): Original image as a numpy array.
        target_size (tuple): Desired image dimensions (width, height).

    Returns:
        tensor: Preprocessed image ready for model input.
    """
    # Resize the input image to fit YOLO model input requirements
    img_resized = cv2.resize(image, target_size)
    
    # Define a transform pipeline: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Apply the transformations and add a batch dimension
    return transform(img_resized).unsqueeze(0)
