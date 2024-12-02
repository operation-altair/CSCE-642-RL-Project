from ultralytics import YOLO
import cv2
import torch

# Assuming the provided predict_image function is already defined here.

# Test predict_image
if __name__ == "__main__":
    # Paths to model and test image
    model_path = "path/to/best.pt"  # Update with your model path
    image_path = "path/to/test_image.jpg"  # Update with your image path

    # Run the predict_image function
    try:
        result = predict_image(model_path, image_path)
        print("Predicted Grid Positions:")
        print(result)
    except Exception as e:
        print("Error during prediction:", e)
