import cv2

def postprocess_segmentation(segmented, kernel_size=7, morphology_operations=2):
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if morphology_operations == 1:
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    elif morphology_operations == 2:
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        processed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    else:
        processed = binary

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    postprocessed = cv2.drawContours(segmented.copy(), contours, -1, (0, 255, 0), 3)
    return postprocessed