import cv2
import numpy as np

def adaptive_preprocess(image, threshold=2, clip_limit=3.0):
    # Convert the image to different color spaces that are useful for segmentation
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Apply CLAHE to each channel to improve the contrast of the image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    hsv_img[:, :, 2] = clahe.apply(hsv_img[:, :, 2])
    ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])

    # Convert back to BGR color space
    enhanced_hsv = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    enhanced_ycrcb = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    # Combine the two enhanced images
    enhanced_img = cv2.addWeighted(enhanced_hsv, 0.5, enhanced_ycrcb, 0.5, 0)

    # Convert to grayscale and apply a bilateral filter to reduce noise while keeping edges sharp
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Apply adaptive thresholding with adjustable threshold
    adaptive_thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, threshold)

    # Canny edge detection to find edges, parameterized by median value
    median_val = np.median(filtered)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    edges = cv2.Canny(adaptive_thresh, lower, upper)

    # Apply morphological operations to close small holes and open clumped objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel, iterations=1)

    return opened_edges