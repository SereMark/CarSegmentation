import cv2

def load_image(image_path):
    """Load an image from the specified file path."""
    return cv2.imread(image_path)

def save_image(image_path, image):
    """Save an image to the specified file path."""
    cv2.imwrite(image_path, image)