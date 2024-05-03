import cv2
import numpy as np

def adaptive_preprocess(image, threshold=2, clip_limit=3.0, color_channel='BGR'):
    enhanced_img = enhance_contrast(image, clip_limit, color_channel)
    filtered_img = apply_bilateral_filter(enhanced_img)
    edges = detect_edges(filtered_img, threshold)
    refined_edges = apply_morphology(edges)
    return refined_edges

def enhance_contrast(image, clip_limit, color_channel):
    if color_channel == 'HSV':
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        hsv_img[:, :, 2] = clahe.apply(hsv_img[:, :, 2])
        enhanced_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    elif color_channel == 'YCrCb':
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
        enhanced_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    else:
        bgr_img = image.copy()
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        for i in range(3):
            bgr_img[:, :, i] = clahe.apply(bgr_img[:, :, i])
        enhanced_img = bgr_img
    return enhanced_img

def apply_bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

def detect_edges(image, threshold):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, threshold)
    median_val = np.median(gray_img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    return cv2.Canny(adaptive_thresh, lower, upper)

def apply_morphology(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel, iterations=1)