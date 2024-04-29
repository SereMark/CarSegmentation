import cv2
import numpy as np

def segment_vehicles(image, preprocessed_img, min_area_ratio=0.005, aspect_ratio_range=(0.2, 3.0), min_solidity=0.5):
    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(preprocessed_img)
    min_area = image.shape[0] * image.shape[1] * min_area_ratio
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area) / hull_area
            if area > min_area and aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and solidity > min_solidity:
                cv2.drawContours(mask, [hull], -1, 255, -1)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented