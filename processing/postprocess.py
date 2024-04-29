import cv2

def postprocess_segmentation(segmented):
    # Convert to grayscale and apply threshold to create a binary image
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Use more aggressive morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # To enhance the visibility of changes, find contours and draw them on the original segmented image
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    postprocessed = cv2.drawContours(segmented.copy(), contours, -1, (0, 255, 0), 3)

    return postprocessed