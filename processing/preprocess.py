import cv2

def adaptive_preprocess(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_v = clahe.apply(v)
    clahe_hsv = cv2.merge([h, s, clahe_v])
    clahe_image = cv2.cvtColor(clahe_hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(clahe_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded