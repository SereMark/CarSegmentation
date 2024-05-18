import random
import tkinter as tk
from tkinter import Label, Scale, OptionMenu, Frame, Button, Checkbutton
import cv2
import numpy as np
from PIL import Image, ImageTk
from random import uniform

def main():
    root = tk.Tk()
    root.title("Vehicle Segmentation Application")
    root.geometry('1200x800')
    image_path = "images/HrpktkpTURBXy83YWViODRjNThkMjExZWQ5ZTRjNzFkOWQxYTM4ZTAyZi5qcGeSlQMABc0S5M0KoJUCzQOlAMLD.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image.")
        return

    control_vars = initialize_control_vars()
    randomize_vars = {var: tk.BooleanVar(value=True) for var in control_vars['control_vars']}
    image_labels = setup_gui(root, control_vars, randomize_vars, image)
    update_processing(control_vars, image_labels, image, randomize_vars)
    root.mainloop()

def initialize_control_vars():
    controls = {
        'noise_controls': [
            ("Noise Intensity", Scale, 'noise_intensity', {'from_': 0, 'to': 0.2, 'resolution': 0.01}),
            ("Noise Type", OptionMenu, 'noise_type', {'options': ['gaussian', 'salt_pepper']})
        ],
        'preprocessing_controls': [
            ("Contrast Clip Limit", Scale, 'contrast_clip_limit', {'from_': 2.0, 'to': 4.0, 'resolution': 0.1}),
            ("Color Channel", OptionMenu, 'color_channel', {'options': ['BGR', 'HSV', 'YCrCb']})
        ],
        'segmentation_controls': [
            ("Segmentation Area Ratio", Scale, 'segmentation_area_ratio', {'from_': 0.001, 'to': 0.01, 'resolution': 0.001}),
            ("Max Area Ratio", Scale, 'max_area_ratio', {'from_': 0.1, 'to': 0.4, 'resolution': 0.01}),
            ("Minimum Aspect Ratio", Scale, 'min_aspect_ratio', {'from_': 0.3, 'to': 1.0, 'resolution': 0.1}),
            ("Maximum Aspect Ratio", Scale, 'max_aspect_ratio', {'from_': 1.0, 'to': 5.0, 'resolution': 0.1}),
            ("Minimum Solidity", Scale, 'min_solidity', {'from_': 0.5, 'to': 1.0, 'resolution': 0.1}),
            ("Vertex Count Threshold", Scale, 'vertex_threshold', {'from_': 4, 'to': 15, 'resolution': 1})
        ],
        'postprocessing_controls': [
            ("Postprocessing Kernel Size", Scale, 'postprocess_kernel_size', {'from_': 3, 'to': 7, 'resolution': 1}),
            ("Morphology Operations", Scale, 'morphology_operations', {'from_': 1, 'to': 5, 'resolution': 1})
        ]
    }
    return {
        'control_vars': {
            'noise_type': tk.StringVar(value='gaussian'),
            'noise_intensity': tk.DoubleVar(value=0.05),
            'contrast_clip_limit': tk.DoubleVar(value=2.5),
            'color_channel': tk.StringVar(value='HSV'),
            'max_area_ratio': tk.DoubleVar(value=0.3),
            'segmentation_area_ratio': tk.DoubleVar(value=0.005),
            'min_aspect_ratio': tk.DoubleVar(value=0.5),
            'max_aspect_ratio': tk.DoubleVar(value=3.0),
            'min_solidity': tk.DoubleVar(value=0.6),
            'vertex_threshold': tk.IntVar(value=10),
            'postprocess_kernel_size': tk.IntVar(value=5),
            'morphology_operations': tk.IntVar(value=2)
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
    image_frame_bottom.grid(row=3, column=0, columnspan=4, sticky='nsew', padx=5, pady=5)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=0)
    root.grid_rowconfigure(3, weight=1)
    root.grid_columnconfigure(0, weight=1)

    titles_top = ["Original", "Noisy", "Preprocessed"]
    titles_bottom = ["Segmented", "Postprocessed"]
    for idx, title in enumerate(titles_top):
        label = Label(image_frame_top, text=title)
        label.grid(row=0, column=idx)
    for idx, title in enumerate(titles_bottom):
        label = Label(image_frame_bottom, text=title)
        label.grid(row=0, column=idx)

    image_labels_top = [
        Label(image_frame_top),
        Label(image_frame_top),
        Label(image_frame_top)
    ]
    image_labels_bottom = [
        Label(image_frame_bottom),
        Label(image_frame_bottom)
    ]
    for idx, img_label in enumerate(image_labels_top):
        img_label.grid(row=1, column=idx, sticky='nsew')
    for idx, img_label in enumerate(image_labels_bottom):
        img_label.grid(row=1, column=idx, sticky='nsew')

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
                option_menu = OptionMenu(frame, control_vars['control_vars'][var_name], *kwargs['options'], command=lambda value, var_name=var_name: on_option_select(var_name, value, control_vars, image_labels, image, randomize_vars))
                option_menu.grid(row=row_index, column=1)
            Checkbutton(frame, text="Randomize", variable=randomize_vars[var_name]).grid(row=row_index, column=2)
        Button(frame, text="Randomize Section", command=lambda fr=section: set_random_values(control_vars, randomize_vars, lambda: update_processing(control_vars, image_labels, image, randomize_vars), fr)).grid(row=len(controls), column=0, columnspan=3)

def on_option_select(var_name, value, control_vars, image_labels, image, randomize_vars):
    control_vars['control_vars'][var_name].set(value)
    update_processing(control_vars, image_labels, image, randomize_vars)

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

def update_processing(control_vars, image_labels, original_image, randomize_vars):
    image = original_image.copy()    
    noisy_img = add_noise(image, control_vars['control_vars']['noise_type'].get(), control_vars['control_vars']['noise_intensity'].get())
    preprocessed_img = preprocess(noisy_img, control_vars['control_vars']['contrast_clip_limit'].get(), control_vars['control_vars']['color_channel'].get())
    segmented_img = segment_vehicles(preprocessed_img, control_vars['control_vars']['segmentation_area_ratio'].get(), 
                                     (control_vars['control_vars']['min_aspect_ratio'].get(), control_vars['control_vars']['max_aspect_ratio'].get()), 
                                     control_vars['control_vars']['min_solidity'].get(), 
                                     control_vars['control_vars']['max_area_ratio'].get(), 
                                     control_vars['control_vars']['vertex_threshold'].get())
    postprocessed_img = postprocess_segmentation(segmented_img, control_vars['control_vars']['postprocess_kernel_size'].get(), control_vars['control_vars']['morphology_operations'].get())

    display_images([original_image, noisy_img, preprocessed_img, segmented_img, postprocessed_img], image_labels)

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
    mean, sigma = 0, intensity * 255
    gauss_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = np.clip(image + gauss_noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_pepper_noise(image, intensity):
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = row * col * ch * intensity
    num_salt = int(amount * s_vs_p)
    num_pepper = int(amount * (1 - s_vs_p))

    coords_salt = (np.random.randint(0, row, num_salt), np.random.randint(0, col, num_salt), np.random.randint(0, ch, num_salt))
    image[coords_salt] = 255

    coords_pepper = (np.random.randint(0, row, num_pepper), np.random.randint(0, col, num_pepper), np.random.randint(0, ch, num_pepper))
    image[coords_pepper] = 0

    return image

def preprocess(image, clip_limit, color_channel):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    channel_map = {
        'HSV': 2,
        'YCrCb': 0
    }
    if color_channel in channel_map:
        image = cv2.cvtColor(image, getattr(cv2, f'COLOR_BGR2{color_channel}'))
        image[:, :, channel_map[color_channel]] = clahe.apply(image[:, :, channel_map[color_channel]])
        processed_img = cv2.cvtColor(image, getattr(cv2, f'COLOR_{color_channel}2BGR'))
    else:
        processed_img = np.stack([clahe.apply(image[:, :, i]) for i in range(3)], axis=-1)

    filtered_img = cv2.bilateralFilter(processed_img, 9, 75, 75)
    _, binary_img = cv2.threshold(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_img

def segment_vehicles(preprocessed_img, min_area_ratio, aspect_ratio_range, min_solidity, max_area_ratio, vertex_threshold):
    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = preprocessed_img.shape[0] * preprocessed_img.shape[1] * min_area_ratio
    max_area = preprocessed_img.shape[0] * preprocessed_img.shape[1] * max_area_ratio

    mask = np.zeros(preprocessed_img.shape, dtype=np.uint8)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and
            solidity > min_solidity and
            is_vehicle_shape(contour, hull, vertex_threshold)):
            cv2.drawContours(mask, [contour], -1, 255, -1)
    
    return mask

def is_vehicle_shape(contour, hull, vertex_threshold):
    if not cv2.isContourConvex(hull):
        return False
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    return len(approx) <= vertex_threshold

def postprocess_segmentation(segmented_img, kernel_size, morphology_operations):
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY) if segmented_img.ndim == 3 else segmented_img
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    morph_ops = [
        lambda img: cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel),
        lambda img: cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel),
        lambda img: cv2.dilate(cv2.erode(img, kernel, iterations=2), kernel, iterations=2),
        lambda img: cv2.erode(cv2.dilate(img, kernel, iterations=2), kernel, iterations=2),
        lambda img: cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2), cv2.MORPH_CLOSE, kernel, iterations=2)
    ]

    processed = morph_ops[min(morphology_operations - 1, len(morph_ops) - 1)](binary)
    return cv2.bitwise_and(segmented_img, segmented_img, mask=processed)

if __name__ == '__main__':
    main()