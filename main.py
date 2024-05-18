import random
import tkinter as tk
from tkinter import Label, Scale, OptionMenu, Frame, Button, Checkbutton
import cv2
from PIL import Image, ImageTk
from random import uniform

import numpy as np

def main():
    root = tk.Tk()
    root.title("Vehicle Segmentation Application")
    root.geometry('1200x800')
    image_path = "images/autopalya-2-1024x576.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image.")
        return

    control_vars = initialize_control_vars()
    randomize_vars = {var: tk.BooleanVar(value=False) for var in control_vars['control_vars']}
    image_labels = setup_gui(root, control_vars, randomize_vars, image)
    update_processing(control_vars, image_labels, image, randomize_vars)

    root.mainloop()

def initialize_control_vars():
    controls = {
        'noise_controls': [
            ("Noise Intensity", Scale, 'noise_intensity', {'from_': 0, 'to': 0.5, 'resolution': 0.01}),
            ("Noise Type", OptionMenu, 'noise_type', {'options': ['gaussian', 'salt_pepper']})
        ],
        'preprocessing_controls': [
            ("Preprocessing Threshold", Scale, 'preprocess_threshold', {'from_': 0.5, 'to': 3.0, 'resolution': 0.1}),
            ("Contrast Clip Limit", Scale, 'contrast_clip_limit', {'from_': 1.0, 'to': 5.0, 'resolution': 0.1}),
            ("Color Channel", OptionMenu, 'color_channel', {'options': ['BGR', 'HSV', 'YCrCb']})
        ],
        'segmentation_controls': [
            ("Segmentation Area Ratio", Scale, 'segmentation_area_ratio', {'from_': 0.001, 'to': 0.02, 'resolution': 0.001}),
            ("Minimum Aspect Ratio", Scale, 'min_aspect_ratio', {'from_': 0.1, 'to': 1.0, 'resolution': 0.1}),
            ("Maximum Aspect Ratio", Scale, 'max_aspect_ratio', {'from_': 1.0, 'to': 10.0, 'resolution': 0.1}),
            ("Minimum Solidity", Scale, 'min_solidity', {'from_': 0.1, 'to': 1.0, 'resolution': 0.1})
        ],
        'postprocessing_controls': [
            ("Postprocessing Kernel Size", Scale, 'postprocess_kernel_size', {'from_': 3, 'to': 10, 'resolution': 1}),
            ("Morphology Operations", Scale, 'morphology_operations', {'from_': 1, 'to': 5, 'resolution': 1})
        ]
    }
    return {
        'control_vars': {
            'noise_type': tk.StringVar(value='gaussian'),
            'noise_intensity': tk.DoubleVar(value=0.00),
            'preprocess_threshold': tk.DoubleVar(value=0.5),
            'contrast_clip_limit': tk.DoubleVar(value=1.0),
            'color_channel': tk.StringVar(value='HSV'),
            'segmentation_area_ratio': tk.DoubleVar(value=0.001),
            'min_aspect_ratio': tk.DoubleVar(value=0.1),
            'max_aspect_ratio': tk.DoubleVar(value=10.0),
            'min_solidity': tk.DoubleVar(value=0.1),
            'postprocess_kernel_size': tk.IntVar(value=3),
            'morphology_operations': tk.IntVar(value=5)
        },
        'controls': controls
    }

def setup_gui(root, control_vars, randomize_vars, image):
    control_frame = Frame(root, bd=2, relief='sunken')
    control_frame.grid(row=0, column=0, columnspan=4, sticky='ew')

    frames = {key: Frame(control_frame, bd=2, relief='sunken') for key in ['noise_frame', 'preprocessing_frame', 'segmentation_frame', 'postprocessing_frame']}
    for i, key in enumerate(frames):
        frames[key].grid(row=0, column=i, padx=5, pady=5)

    image_frame_top, image_frame_bottom = Frame(root, bd=2, relief='sunken'), Frame(root, bd=2, relief='sunken')
    image_frame_top.grid(row=1, column=0, columnspan=4, sticky='nsew', padx=5, pady=5)
    image_frame_bottom.grid(row=2, column=0, columnspan=4, sticky='nsew', padx=5, pady=5)

    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)
    root.grid_columnconfigure(0, weight=1)

    image_labels_top = [Label(image_frame_top) for _ in range(3)]
    image_labels_bottom = [Label(image_frame_bottom) for _ in range(2)]

    for img_label in image_labels_top + image_labels_bottom:
        img_label.pack(side='left', expand=True, fill='both')

    image_labels = image_labels_top + image_labels_bottom

    setup_controls(frames, control_vars, randomize_vars, lambda: update_processing(control_vars, image_labels, image, randomize_vars), image_labels, image)
    return image_labels

def setup_controls(frames, control_vars, randomize_vars, update_function, image_labels, image):
    frame_keys = {
        'noise_controls': 'noise_frame',
        'preprocessing_controls': 'preprocessing_frame',
        'segmentation_controls': 'segmentation_frame',
        'postprocessing_controls': 'postprocessing_frame'
    }

    for section, controls in control_vars['controls'].items():
        frame = frames[frame_keys[section]]
        for row_index, (label_text, widget, var_name, kwargs) in enumerate(controls):
            Label(frame, text=label_text).grid(row=row_index, column=0)
            if widget == Scale:
                scale_widget = Scale(frame, variable=control_vars['control_vars'][var_name], orient='horizontal', **kwargs)
                scale_widget.grid(row=row_index, column=1)
                scale_widget.bind("<ButtonRelease-1>", lambda event, var_name=var_name: update_processing(control_vars, image_labels, image, randomize_vars))
            elif widget == OptionMenu:
                option_menu = OptionMenu(frame, control_vars['control_vars'][var_name], *kwargs['options'])
                option_menu.grid(row=row_index, column=1)
            Checkbutton(frame, text="Randomize", variable=randomize_vars[var_name]).grid(row=row_index, column=2)
        Button(frame, text="Randomize Section", command=lambda fr=section: set_random_values(control_vars, randomize_vars, lambda: update_processing(control_vars, image_labels, image, randomize_vars), fr)).grid(row=len(controls), column=0, columnspan=3)

def set_random_values(control_vars, randomize_vars, update_function, section=None):
    triggered = False
    if section:
        for label_text, control_type, var_name, kwargs in control_vars['controls'][section]:
            if randomize_vars[var_name].get():
                if control_type == OptionMenu:
                    new_value = random.choice(kwargs['options'])
                elif control_type == Scale:
                    range_min, range_max = kwargs['from_'], kwargs['to']
                    new_value = uniform(range_min, range_max)
                control_vars['control_vars'][var_name].set(new_value)
                triggered = True
    if triggered:
        update_function()

def update_processing(control_vars, image_labels, image, randomize_vars):
    noisy_img = add_noise(image, control_vars['control_vars']['noise_type'].get(), control_vars['control_vars']['noise_intensity'].get())
    preprocessed_img = preprocess(noisy_img, control_vars['control_vars']['preprocess_threshold'].get(), control_vars['control_vars']['contrast_clip_limit'].get(), control_vars['control_vars']['color_channel'].get())
    segmented_img = segment_vehicles(preprocessed_img, control_vars['control_vars']['segmentation_area_ratio'].get(), (control_vars['control_vars']['min_aspect_ratio'].get(), control_vars['control_vars']['max_aspect_ratio'].get()), control_vars['control_vars']['min_solidity'].get())
    postprocessed_img = postprocess_segmentation(segmented_img, control_vars['control_vars']['postprocess_kernel_size'].get(), control_vars['control_vars']['morphology_operations'].get())

    display_images([image, noisy_img, preprocessed_img, segmented_img, postprocessed_img], image_labels)

def display_images(images, image_labels):
    max_width, max_height = 800, 600
    for idx, img in enumerate(images):
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if img.ndim == 3 else cv2.COLOR_GRAY2RGB)
            height, width = img.shape[:2]
            scaling_factor = min(max_width / width, max_height / height)
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            img_pil = ImageTk.PhotoImage(image=Image.fromarray(img))
            if idx < len(image_labels):
                image_labels[idx].configure(image=img_pil)
                image_labels[idx].image = img_pil
            else:
                print(f"Warning: More images ({len(images)}) than labels ({len(image_labels)}) provided.")
        
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

def preprocess(image, threshold, clip_limit, color_channel):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    if color_channel == 'HSV':
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_img[:, :, 2] = clahe.apply(hsv_img[:, :, 2])
        processed_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    elif color_channel == 'YCrCb':
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
        processed_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    else:
        processed_img = image.copy()
        for i in range(3):
            processed_img[:, :, i] = clahe.apply(image[:, :, i])

    filtered_img = cv2.bilateralFilter(processed_img, 9, 75, 75)
    gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_img

def segment_vehicles(preprocessed_img, min_area_ratio, aspect_ratio_range, min_solidity):
    if preprocessed_img.ndim != 2:
        raise ValueError("Preprocessed image must be a single-channel binary image.")

    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = preprocessed_img.shape[0] * preprocessed_img.shape[1] * min_area_ratio

    mask = np.zeros(preprocessed_img.shape, dtype=np.uint8)
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

    return mask

def postprocess_segmentation(segmented_img, kernel_size, morphology_operations):
    if segmented_img.ndim == 3 and segmented_img.shape[2] == 3:
        gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = segmented_img

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

    return cv2.bitwise_and(segmented_img, segmented_img, mask=processed)

def exclude_shadows(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    _, shadow_mask = cv2.threshold(v_channel, 50, 255, cv2.THRESH_BINARY)
    refined_mask = cv2.bitwise_and(mask, shadow_mask)
    return refined_mask

if __name__ == '__main__':
    main()