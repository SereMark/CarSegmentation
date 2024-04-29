import cv2

def load_image(path):
    """ Load an image from a specified path. """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"The image at path {path} could not be loaded.")
    return image

def save_image(path, image):
    """ Save an image to a specified path. """
    cv2.imwrite(path, image)