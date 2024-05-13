import tkinter as tk
from tkinter import Checkbutton, Label, Scale, OptionMenu, Frame
import cv2
from PIL import Image, ImageTk
from random import uniform
import numpy as np

# Application Title
def main():
    root = tk.Tk()
    root.title("Vehicle Segmentation Application")
    root.geometry('1200x800')
    image_path = "images/autopalya-2-1024x576.jpg"
    image = cv2.imread(image_path)

    control_vars = initialize_control_vars()
    randomize_vars = {var: tk.BooleanVar(value=True) for var in control_vars if var not in {'controls', 'noise_type', 'color_channel'}}
    image_labels = setup_gui(root, control_vars, randomize_vars, image)
    update_processing(control_vars, image_labels, image, randomize_vars)

    root.mainloop()

# Initialize control variables
def initialize_control_vars():
    controls = [
        ("Noise Intensity", Scale, 'noise_intensity', {'from_': 0, 'to': 0.5, 'resolution': 0.01}),
        ("Preprocessing Threshold", Scale, 'preprocess_threshold', {'from_': 0.5, 'to': 3.0, 'resolution': 0.1}),
        ("Contrast Clip Limit", Scale, 'contrast_clip_limit', {'from_': 1.0, 'to': 5.0, 'resolution': 0.1}),
        ("Segmentation Area Ratio", Scale, 'segmentation_area_ratio', {'from_': 0.001, 'to': 0.02, 'resolution': 0.001}),
        ("Minimum Aspect Ratio", Scale, 'min_aspect_ratio', {'from_': 0.1, 'to': 1.0, 'resolution': 0.1}),
        ("Maximum Aspect Ratio", Scale, 'max_aspect_ratio', {'from_': 1.0, 'to': 10.0, 'resolution': 0.1}),
        ("Minimum Solidity", Scale, 'min_solidity', {'from_': 0.1, 'to': 1.0, 'resolution': 0.1}),
        ("Postprocessing Kernel Size", Scale, 'postprocess_kernel_size', {'from_': 3, 'to': 10, 'resolution': 1}),
        ("Morphology Operations", Scale, 'morphology_operations', {'from_': 1, 'to': 5, 'resolution': 1})
    ]
    return {
        'noise_type': tk.StringVar(value='gaussian'),
        'noise_intensity': tk.DoubleVar(value=0.02),
        'preprocess_threshold': tk.DoubleVar(value=1.5),
        'contrast_clip_limit': tk.DoubleVar(value=3.0),
        'color_channel': tk.StringVar(value='HSV'),
        'segmentation_area_ratio': tk.DoubleVar(value=0.01),
        'min_aspect_ratio': tk.DoubleVar(value=0.75),
        'max_aspect_ratio': tk.DoubleVar(value=4.0),
        'min_solidity': tk.DoubleVar(value=0.3),
        'postprocess_kernel_size': tk.IntVar(value=5),
        'morphology_operations': tk.IntVar(value=5),
        'controls': controls
    }

# Set up GUI layout and controls
def setup_gui(root, control_vars, randomize_vars, image):
    image_frame_top = Frame(root)
    image_frame_top.grid(row=0, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
    image_frame_bottom = Frame(root)
    image_frame_bottom.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)

    control_frame = Frame(root)
    control_frame.grid(row=1, column=2, sticky='nsew', padx=5, pady=5)

    image_labels = [Label(image_frame_top) for _ in range(3)] + [Label(image_frame_bottom) for _ in range(2)]
    for img_label in image_labels:
        img_label.pack(side='left', expand=True, fill='both', padx=5, pady=5)

    setup_controls(control_frame, control_vars, randomize_vars, lambda: update_processing(control_vars, image_labels, image, randomize_vars))
    return image_labels

# Setup control widgets
def setup_controls(control_frame, control_vars, randomize_vars, update_function):
    Label(control_frame, text="Noise Type").grid(row=0, column=0, columnspan=2)
    OptionMenu(control_frame, control_vars['noise_type'], 'gaussian', 'salt_pepper', command=lambda _: update_function()).grid(row=0, column=2, columnspan=2)

    Label(control_frame, text="Color Channel").grid(row=1, column=0, columnspan=2)
    OptionMenu(control_frame, control_vars['color_channel'], 'BGR', 'HSV', 'YCrCb', command=lambda _: update_function()).grid(row=1, column=2, columnspan=2)

    row_index = 2
    for label_text, widget, var, kwargs in control_vars['controls']:
        Label(control_frame, text=label_text).grid(row=row_index, column=0)
        widget(control_frame, variable=control_vars[var], orient='horizontal', command=lambda _: update_function(), **kwargs).grid(row=row_index, column=1)
        Checkbutton(control_frame, text="Randomize", variable=randomize_vars[var]).grid(row=row_index, column=2)
        row_index += 1

    tk.Button(control_frame, text="Randomize All", command=lambda: set_random_values(control_vars, randomize_vars, update_function)).grid(row=row_index, column=0, columnspan=3)

# Process image and display
def update_processing(control_vars, image_labels, image, randomize_vars):
    noisy_img = add_noise(image, control_vars['noise_type'].get(), control_vars['noise_intensity'].get())
    preprocessed_img = adaptive_preprocess(noisy_img, control_vars['preprocess_threshold'].get(), control_vars['contrast_clip_limit'].get(), control_vars['color_channel'].get())
    segmented_img = segment_vehicles(preprocessed_img, control_vars['segmentation_area_ratio'].get(), (control_vars['min_aspect_ratio'].get(), control_vars['max_aspect_ratio'].get()), control_vars['min_solidity'].get())
    postprocessed_img = postprocess_segmentation(segmented_img, control_vars['postprocess_kernel_size'].get(), control_vars['morphology_operations'].get())
    display_images([image, noisy_img, preprocessed_img, segmented_img, postprocessed_img], image_labels)

# Display updated images on GUI
def display_images(images, image_labels):
    for idx, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if img.ndim == 2 else cv2.COLOR_BGR2RGB)
        new_width = 835
        aspect_ratio = img.shape[1] / img.shape[0]
        new_height = int(new_width / aspect_ratio)
        img = cv2.resize(img, (new_width, new_height))
        img_pil = ImageTk.PhotoImage(image=Image.fromarray(img))
        img_label = image_labels[idx]
        img_label.configure(image=img_pil)
        img_label.image = img_pil

# Randomize control variables
def set_random_values(control_vars, randomize_vars, update_function):
    for var in randomize_vars:
        if randomize_vars[var].get():
            control_vars[var].set(uniform(control_vars[var]._from, control_vars[var]._to))
    update_function()

# Noise addition
def add_noise(image, noise_type='gaussian', intensity=0.1):
    noise_funcs = {
        'gaussian': lambda img: add_gaussian_noise(img, intensity),
        'salt_pepper': lambda img: add_salt_pepper_noise(img, intensity)
    }
    return noise_funcs[noise_type](image)

def add_gaussian_noise(image, intensity):
    row, col, ch = image.shape
    mean = 0
    sigma = intensity * 255
    gauss_noise = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
    return np.clip(image + gauss_noise, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, intensity):
    row, col, ch = image.shape
    s_vs_p = 0.5
    num_salt = int(row * col * ch * intensity * s_vs_p)
    num_pepper = int(row * col * ch * intensity * (1 - s_vs_p))

    salt_coords = (np.random.randint(0, row, num_salt), np.random.randint(0, col, num_salt), np.random.randint(0, ch, num_salt))
    pepper_coords = (np.random.randint(0, row, num_pepper), np.random.randint(0, col, num_pepper), np.random.randint(0, ch, num_pepper))

    out_image = image.copy()
    out_image[salt_coords] = 255
    out_image[pepper_coords] = 0
    return out_image

# Image preprocessing
def adaptive_preprocess(image, threshold, clip_limit, color_channel):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    if color_channel == 'HSV':
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_img[:, :, 2] = clahe.apply(hsv_img[:, :, 2])
        image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    elif color_channel == 'YCrCb':
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
        image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    else:
        for i in range(3):
            image[:, :, i] = clahe.apply(image[:, :, i])

    filtered_img = cv2.bilateralFilter(image, 9, 75, 75)
    gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, threshold)
    median_val = np.median(gray_img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    edges = cv2.Canny(adaptive_thresh, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel, iterations=1)

# Vehicle segmentation
def segment_vehicles(preprocessed_img, min_area_ratio, aspect_ratio_range, min_solidity):
    if preprocessed_img.ndim == 2:
        preprocessed_img_color = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
    else:
        preprocessed_img_color = preprocessed_img

    blurred_img = cv2.GaussianBlur(preprocessed_img, (5, 5), 0)
    contours, _ = cv2.findContours(blurred_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = preprocessed_img.shape[0] * preprocessed_img.shape[1] * min_area_ratio

    mask = np.zeros_like(preprocessed_img, dtype=np.uint8)
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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_no_shadows = exclude_shadows(preprocessed_img_color, mask)
    return cv2.bitwise_and(preprocessed_img, preprocessed_img, mask=mask_no_shadows)

# Post-process segmented image
def postprocess_segmentation(segmented, kernel_size, morphology_operations):
    if segmented.ndim == 3 and segmented.shape[2] == 3:
        gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    else:
        gray = segmented

    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    processed = binary
    if morphology_operations == 1:
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    elif morphology_operations == 2:
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    elif morphology_operations == 3:
        processed = cv2.erode(binary, kernel, iterations=2)
        processed = cv2.dilate(processed, kernel, iterations=2)
    elif morphology_operations == 4:
        processed = cv2.dilate(binary, kernel, iterations=2)
        processed = cv2.erode(processed, kernel, iterations=2)
    elif morphology_operations == 5:
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=2)

    return cv2.bitwise_and(segmented, segmented, mask=processed)

# Exclude shadows from mask
def exclude_shadows(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    _, shadow_mask = cv2.threshold(v_channel, 50, 255, cv2.THRESH_BINARY)
    refined_mask = cv2.bitwise_and(mask, shadow_mask)
    return refined_mask

if __name__ == '__main__':
    main()