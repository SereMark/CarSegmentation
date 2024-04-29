import cv2

def adaptive_preprocess(image):
    """ Adaptive preprocessing of the image based on lighting conditions using HSV and histogram equalization. """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    equalized_v = cv2.equalizeHist(v)
    equalized_hsv = cv2.merge([h, s, equalized_v])
    equalized_image = cv2.cvtColor(equalized_hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, adaptive_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return adaptive_thresh