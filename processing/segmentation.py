import cv2
import numpy as np

def segment_vehicles(image, preprocessed_img):
    """ Segment vehicles from the image based on preprocessed input. """
    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(preprocessed_img)
    min_area = image.shape[0] * image.shape[1] * 0.01  # Threshold area
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented