import cv2

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"The image at path {path} could not be loaded.")
    return image

def save_image(path, image):
    cv2.imwrite(path, image)