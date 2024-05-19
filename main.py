import random
import tkinter as tk
from tkinter import Label, Scale, OptionMenu, Frame, Button, Checkbutton
import cv2
import numpy as np
from PIL import Image, ImageTk
from random import uniform

def main():
    # Initialize the main window using Tkinter.
    root = tk.Tk()
    root.title("Vehicle Segmentation Application")
    root.geometry('2560x1440')  # Set the size of the main window.

    # Load an image for processing.
    image_path = "images/autopalya-2-1024x576.jpg"
    image = cv2.imread(image_path)  # Read the image using OpenCV.

    # Initialize control variables for the GUI.
    control_vars = initialize_control_vars()

    # Create a dictionary to store Boolean variables for randomization.
    randomize_vars = {var: tk.BooleanVar(value=True) for var in control_vars['control_vars']}

    # Set up the GUI components and link them to control variables.
    image_labels = setup_gui(root, control_vars, randomize_vars, image)

    # Initialize the previous state for comparison in updates.
    control_vars['previous_state'] = {
        'noise_type': control_vars['control_vars']['noise_type'].get(),
        'noise_intensity': control_vars['control_vars']['noise_intensity'].get(),
        'noisy_img': None
    }

    # Process the image initially with default settings.
    update_processing(control_vars, image_labels, image, randomize_vars)

    # Start the Tkinter event loop.
    root.mainloop()

def initialize_control_vars():
    # Define controls for various parameters in the GUI with default values and ranges.
    controls = {
        'noise_controls': [("Noise Intensity", Scale, 'noise_intensity', {'from_': 0, 'to': 0.1, 'resolution': 0.01}),
                           ("Noise Type", OptionMenu, 'noise_type', {'options': ['gaussian', 'salt_pepper']})],
        'preprocessing_controls': [("Contrast Clip Limit", Scale, 'contrast_clip_limit', {'from_': 3.0, 'to': 5.0, 'resolution': 0.1}),
                                   ("Tile Grid Size", Scale, 'tile_grid_size', {'from_': 2, 'to': 8, 'resolution': 1}),
                                   ("Color Channel", OptionMenu, 'color_channel', {'options': ['BGR', 'HSV', 'YCrCb']})],
        'segmentation_controls': [("Segmentation Area Ratio", Scale, 'segmentation_area_ratio', {'from_': 0.0005, 'to': 0.01, 'resolution': 0.0005}),
                                  ("Max Area Ratio", Scale, 'max_area_ratio', {'from_': 0.2, 'to': 0.5, 'resolution': 0.05}),
                                  ("Minimum Aspect Ratio", Scale, 'min_aspect_ratio', {'from_': 0.2, 'to': 1.0, 'resolution': 0.1}),
                                  ("Maximum Aspect Ratio", Scale, 'max_aspect_ratio', {'from_': 1.0, 'to': 5.5, 'resolution': 0.1}),
                                  ("Minimum Solidity", Scale, 'min_solidity', {'from_': 0.4, 'to': 1.0, 'resolution': 0.1}),
                                  ("Vertex Count Threshold", Scale, 'vertex_threshold', {'from_': 4, 'to': 20, 'resolution': 1})],
        'postprocessing_controls': [("Postprocessing Kernel Size", Scale, 'postprocess_kernel_size', {'from_': 5, 'to': 9, 'resolution': 1}),
                                    ("Morphology Operations", Scale, 'morphology_operations', {'from_': 1, 'to': 7, 'resolution': 1})]
    }
    return {
        'control_vars': {
            'noise_type': tk.StringVar(value='salt_pepper'),
            'noise_intensity': tk.DoubleVar(value=0.09),
            'contrast_clip_limit': tk.DoubleVar(value=5.0),
            'tile_grid_size': tk.IntVar(value=2),
            'color_channel': tk.StringVar(value='BGR'),
            'segmentation_area_ratio': tk.DoubleVar(value=0.0005),
            'max_area_ratio': tk.DoubleVar(value=0.50),
            'min_aspect_ratio': tk.DoubleVar(value=0.7),
            'max_aspect_ratio': tk.DoubleVar(value=3.7),
            'min_solidity': tk.DoubleVar(value=0.5),
            'vertex_threshold': tk.IntVar(value=12),
            'postprocess_kernel_size': tk.IntVar(value=9),
            'morphology_operations': tk.IntVar(value=7)
        },
        'controls': controls
    }

def setup_gui(root, control_vars, randomize_vars, image):
    # Create frames for organizing GUI controls and image displays.
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

    # Create labels for different image processing stages.
    titles_top = ["Original", "Noisy", "Preprocessed"]
    titles_bottom = ["Segmented", "Postprocessed"]
    for idx, title in enumerate(titles_top):
        label = Label(image_frame_top, text=title)
        label.grid(row=0, column=idx)
    for idx, title in enumerate(titles_bottom):
        label = Label(image_frame_bottom, text=title)
        label.grid(row=0, column=idx)

    # Placeholders for images at each stage of processing.
    image_labels_top = [Label(image_frame_top), Label(image_frame_top), Label(image_frame_top)]
    image_labels_bottom = [Label(image_frame_bottom), Label(image_frame_bottom)]
    for idx, img_label in enumerate(image_labels_top):
        img_label.grid(row=1, column=idx, sticky='nsew')
    for idx, img_label in enumerate(image_labels_bottom):
        img_label.grid(row=1, column=idx, sticky='nsew')
    image_labels = image_labels_top + image_labels_bottom

    # Attach control widgets to the GUI.
    setup_controls(frames, control_vars, randomize_vars, lambda: update_processing(control_vars, image_labels, image, randomize_vars), image_labels, image)
    Button(control_frame, text="Randomize All", command=lambda: randomize_all(control_vars, randomize_vars, lambda: update_processing(control_vars, image_labels, image, randomize_vars))).grid(row=1, column=0, columnspan=4)
    return image_labels

def setup_controls(frames, control_vars, randomize_vars, update_function, image_labels, image):
    # Mapping of control sections to their respective frames.
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
    # Update control variable when an option is selected and reprocess image.
    control_vars['control_vars'][var_name].set(value)
    update_processing(control_vars, image_labels, image, randomize_vars)

def set_random_values(control_vars, randomize_vars, update_function, section=None):
    # Randomize selected control variables within the specified section.
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

def randomize_all(control_vars, randomize_vars, update_function):
    # Randomize all control variables that are enabled for randomization.
    triggered = False
    for section in control_vars['controls'].keys():
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
    # Retrieve current settings for noise type and intensity.
    current_noise_type = control_vars['control_vars']['noise_type'].get()
    current_noise_intensity = control_vars['control_vars']['noise_intensity'].get()

    # Determine if the noise settings have changed since last update.
    noise_changed = (current_noise_type != control_vars['previous_state']['noise_type'] or
                     current_noise_intensity != control_vars['previous_state']['noise_intensity'])

    # If noise settings changed, apply new noise to a copy of the original image.
    if noise_changed or control_vars['previous_state']['noisy_img'] is None:
        noisy_img = add_noise(original_image.copy(), current_noise_type, current_noise_intensity)
        control_vars['previous_state']['noise_type'] = current_noise_type
        control_vars['previous_state']['noise_intensity'] = current_noise_intensity
        control_vars['previous_state']['noisy_img'] = noisy_img
    else:
        noisy_img = control_vars['previous_state']['noisy_img']

    # Preprocess the noisy image using contrast, color channels, and tile grid settings.
    preprocessed_img = preprocess(noisy_img, control_vars['control_vars']['contrast_clip_limit'].get(), control_vars['control_vars']['color_channel'].get(), control_vars['control_vars']['tile_grid_size'].get())

    # Segment vehicles from the preprocessed image using various geometric properties.
    segmented_img = segment_vehicles(preprocessed_img, control_vars['control_vars']['segmentation_area_ratio'].get(), 
                                     (control_vars['control_vars']['min_aspect_ratio'].get(), control_vars['control_vars']['max_aspect_ratio'].get()), 
                                     control_vars['control_vars']['min_solidity'].get(), 
                                     control_vars['control_vars']['max_area_ratio'].get(), 
                                     control_vars['control_vars']['vertex_threshold'].get())

    # Postprocess the segmented image to improve visual quality.
    postprocessed_img = postprocess_segmentation(segmented_img, control_vars['control_vars']['postprocess_kernel_size'].get(), control_vars['control_vars']['morphology_operations'].get())

    # Display all processed images in the GUI.
    display_images([original_image, noisy_img, preprocessed_img, segmented_img, postprocessed_img], image_labels)

def display_images(images, image_labels):
    # Resize and convert images for display, and update the GUI components with the new images.
    max_width, max_height = 800, 600
    for idx, img in enumerate(images):
        if img is not None:
            # Convert image color space from BGR to RGB for display.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if img.ndim == 3 else cv2.COLOR_GRAY2RGB)
            # Calculate the scaling factor to maintain aspect ratio.
            height, width = img.shape[:2]
            scaling_factor = min(max_width / width, max_height / height)
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            # Resize image using the calculated scaling factor.
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            # Convert the OpenCV image to a PIL image, then to ImageTk format.
            img_pil = ImageTk.PhotoImage(image=Image.fromarray(img))
            # Update the label for this image with the new image data.
            image_labels[idx].configure(image=img_pil)
            image_labels[idx].image = img_pil
        
def add_noise(image, noise_type='gaussian', intensity=0.1):
    # Select the noise function based on the noise type and apply it to the image.
    noise_funcs = {
        'gaussian': lambda img: add_gaussian_noise(img, intensity),
        'salt_pepper': lambda img: add_salt_pepper_noise(img, intensity)
    }
    return noise_funcs[noise_type](image)

def add_gaussian_noise(image, intensity):
    # Generate Gaussian noise with mean=0 and standard deviation proportional to intensity.
    mean, sigma = 0, intensity * 255
    gauss_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    # Add noise to the original image and clip the values to valid color range.
    noisy_image = np.clip(image + gauss_noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_pepper_noise(image, intensity):
    # Add salt and pepper noise to the image by randomly changing pixel values to white or black.
    row, col, ch = image.shape
    s_vs_p = 0.5  # Ratio of salt to pepper.
    amount = row * col * ch * intensity  # Total amount of noise pixels.
    num_salt = int(amount * s_vs_p)
    num_pepper = int(amount * (1 - s_vs_p))
    # Randomly select pixel coordinates for salt noise (white).
    coords_salt = (np.random.randint(0, row, num_salt), np.random.randint(0, col, num_salt), np.random.randint(0, ch, num_salt))
    image[coords_salt] = 255  # Set salt pixels to white.
    # Randomly select pixel coordinates for pepper noise (black).
    coords_pepper = (np.random.randint(0, row, num_pepper), np.random.randint(0, col, num_pepper), np.random.randint(0, ch, num_pepper))
    image[coords_pepper] = 0  # Set pepper pixels to black.
    return image

def preprocess(image, clip_limit, color_channel, tile_grid_size):
    # Set the tile size for CLAHE based on the input tile grid size.
    tile_size = (tile_grid_size, tile_grid_size)
    
    # Initialize the CLAHE object with the specified clip limit and tile grid size.
    # CLAHE is an advanced form of histogram equalization that is adaptive to small regions of the image.
    # It improves the contrast of the image by transforming the values of pixels such that the histogram
    # of the output image is more uniformly distributed.
    # The clip limit controls the contrast amplification limit to avoid amplifying noise.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    if color_channel in ['HSV', 'YCrCb']:
        # Convert the image from BGR to the specified color channel.
        # HSV and YCrCb are color spaces that separate image luminance from color information,
        # which can be more effective for processing than the traditional BGR color space.
        image = cv2.cvtColor(image, getattr(cv2, f'COLOR_BGR2{color_channel}'))

        # Apply CLAHE to the first two channels of the color space, which are typically the channels
        # that carry color information. This enhances the local contrast in these channels.
        for i in range(2):
            image[:, :, i] = clahe.apply(image[:, :, i])

        # Convert the processed image back to BGR color space for further processing and display.
        processed_img = cv2.cvtColor(image, getattr(cv2, f'COLOR_{color_channel}2BGR'))
    else:
        # Apply CLAHE directly to the BGR image if no specific color channel conversion is required.
        processed_img = image

    # Apply a bilateral filter to the processed image.
    # A bilateral filter is a non-linear, edge-preserving, and noise-reducing smoothing filter.
    # It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels.
    # The weights depend not only on the Euclidean distance of pixels but also on the radiometric differences
    # (e.g., color intensity, depth distance), which preserves sharp edges by systematically looping through each pixel
    # and adjusting weights to the adjacent pixels accordingly.
    filtered_img = cv2.bilateralFilter(processed_img, 9, 75, 75)

    # Apply adaptive thresholding to convert the image to a binary format.
    # This step is crucial for separating objects (vehicles) from the background.
    # Adaptive thresholding determines the threshold for a pixel based on a small region around it,
    # allowing for different thresholds for different regions of the same image,
    # which provides better results for images with varying illumination.
    adaptive_thresh = cv2.adaptiveThreshold(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return adaptive_thresh

def segment_vehicles(preprocessed_img, min_area_ratio, aspect_ratio_range, min_solidity, max_area_ratio, vertex_threshold):
    # Detect all external contours in the preprocessed image.
    # Contours are detected using the RETR_EXTERNAL mode to retrieve only the extreme outer contours,
    # which is useful for detecting individual objects.
    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate minimum and maximum allowable area for a contour to be considered a potential vehicle.
    # This is based on the area ratios provided, which helps in filtering out noise and irrelevant objects.
    min_area = preprocessed_img.shape[0] * preprocessed_img.shape[1] * min_area_ratio
    max_area = preprocessed_img.shape[0] * preprocessed_img.shape[1] * max_area_ratio
    
    # Create a mask to draw the valid vehicle contours.
    mask = np.zeros(preprocessed_img.shape, dtype=np.uint8)

    for contour in contours:
        # Calculate the area of each contour.
        area = cv2.contourArea(contour)
        # Skip contours that are too small or too large based on predefined thresholds.
        if area < min_area or area > max_area:
            continue
        
        # Compute the bounding rectangle of the contour to get the aspect ratio (width/height).
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # Compute the convex hull of the contour and its area to calculate solidity.
        # Solidity is the ratio of contour area to its convex hull area. It measures the "solidity" of the shape,
        # which helps in distinguishing irregular shapes from regular ones (like vehicles).
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Validate the contour based on aspect ratio, solidity, and a simplified shape criterion.
        # This checks if the shape of the contour is within acceptable limits, resembling that of typical vehicles.
        if (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and solidity > min_solidity and is_vehicle_shape(contour, hull, vertex_threshold)):
            # Draw the contour on the mask if it qualifies as a potential vehicle.
            cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask

def is_vehicle_shape(contour, hull, vertex_threshold):
    # Check if the contour can be approximated to have fewer than a set number of vertices (simplification).
    if not cv2.isContourConvex(hull):  # Ignore non-convex contours.
        return False
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    return len(approx) <= vertex_threshold  # Check if the number of vertices is less than the threshold.

def postprocess_segmentation(segmented_img, kernel_size, morphology_operations):
    # Convert the segmented image to grayscale for morphological processing.
    # This conversion simplifies the image to one channel, making it suitable for binary thresholding and morphological processing.
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY) if segmented_img.ndim == 3 else segmented_img
    
    # Apply a binary threshold to create a binary image.
    # This step converts the grayscale image to a binary image where the pixels will either be 0 (black) or 255 (white).
    # The threshold value is set to 1, which means all pixels with intensity greater than 1 will be set to 255.
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Create a structural element for morphological operations.
    # An elliptical kernel is chosen here, which is effective for ensuring smooth, rounded edges in the output.
    # The size of the kernel is determined by the 'kernel_size' parameter, influencing how much the image is processed.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Define a list of possible morphological operations to apply.
    # These operations are defined as lambda functions for compactness and are applied conditionally based on the input settings.
    morph_ops = [
        lambda img: img,  # No operation: simply returns the image as is, useful for comparison.
        lambda img: cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel),  # Opening: Erosion followed by dilation. It is useful for removing noise.
        lambda img: cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel),  # Closing: Dilation followed by erosion. It is useful for closing small holes and gaps.
        lambda img: cv2.dilate(cv2.erode(img, kernel, iterations=2), kernel, iterations=2),  # First erode then dilate, both twice. This sequence enhances the definition of larger structures.
        lambda img: cv2.erode(cv2.dilate(img, kernel, iterations=2), kernel, iterations=2),  # First dilate then erode, both twice. This sequence can help in defining fine structures.
        lambda img: cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2), cv2.MORPH_CLOSE, kernel, iterations=2)  # Apply opening followed by closing, each twice. This combination helps in refining the segmentation results by smoothing the contours and closing small holes.
    ]
    
    # Apply the selected morphological operation based on the current setting.
    # 'morphology_operations' parameter decides which operation from 'morph_ops' list to apply.
    # If the selected index is out of range, it defaults to applying no operation.
    if 0 <= morphology_operations < len(morph_ops):
        processed = morph_ops[morphology_operations](binary)
    else:
        processed = binary
    
    # Combine the original segmented image with the processed mask to finalize the postprocessing.
    # This step uses the mask created by morphological operations to keep only the relevant parts of the segmented image,
    # effectively enhancing the segmentation by applying the mask.
    return cv2.bitwise_and(segmented_img, segmented_img, mask=processed)

if __name__ == '__main__':
    main()