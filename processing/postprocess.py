import cv2

def postprocess_segmentation(segmented):
    """ Postprocess segmented image to refine the results. """
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY) if len(segmented.shape) == 3 else segmented
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)
    final = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
    postprocessed = cv2.bitwise_and(segmented, segmented, mask=final)
    return postprocessed