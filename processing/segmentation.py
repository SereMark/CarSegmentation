import cv2
import numpy as np

def exclude_shadows(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_mean = np.mean(v)
    s_mean = np.mean(s)
    shadow_threshold_s = max(40, s_mean * 0.5)
    shadow_threshold_v = max(75, v_mean * 0.5)
    shadow_mask = (s < shadow_threshold_s) & (v < shadow_threshold_v)
    return cv2.bitwise_and(mask, mask, mask=~shadow_mask.astype(np.uint8))

def segment_vehicles(image, preprocessed_img, min_area_ratio=0.005, aspect_ratio_range=(0.2, 5.0), min_solidity=0.5):
    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(preprocessed_img, dtype=np.uint8)
    min_area = image.shape[0] * image.shape[1] * min_area_ratio

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and solidity > min_solidity:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    mask_no_shadows = exclude_shadows(image, mask)
    segmented = cv2.bitwise_and(image, image, mask=mask_no_shadows)
    return segmented