import tkinter as tk
from tkinter import Label, Scale, OptionMenu, Frame
import cv2
from PIL import Image, ImageTk
from utils.noise import add_noise
from processing.preprocess import adaptive_preprocess
from processing.segmentation import segment_vehicles
from processing.postprocess import postprocess_segmentation

def main():
    def update_processing(*args):
        # Update image processing with more parameters if needed
        noisy_img = add_noise(image.copy(), noise_type.get(), noise_intensity.get())
        preprocessed_img = adaptive_preprocess(
            noisy_img.copy(),
            threshold=preprocess_threshold.get(),
            clip_limit=contrast_clip_limit.get()
        )
        segmented_img = segment_vehicles(
            image.copy(),
            preprocessed_img.copy(),
            min_area_ratio=segmentation_area_ratio.get(),
            aspect_ratio_range=(min_aspect_ratio.get(), max_aspect_ratio.get()),
            min_solidity=min_solidity.get()
        )
        postprocessed_img = postprocess_segmentation(
            segmented_img.copy(),
            kernel_size=postprocess_kernel_size.get()
        )
        
        # Display the processed images
        display_images([image, noisy_img, preprocessed_img, segmented_img, postprocessed_img])

    def display_images(images):
        # Update the image labels with the new images
        for idx, img_label in enumerate(image_labels):
            img = images[idx]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if img.ndim == 2 else cv2.COLOR_BGR2RGB)
            
            # Resize the image to fit the window
            new_width = 800
            aspect_ratio = img.shape[1] / img.shape[0]  # width / height
            new_height = int(new_width / aspect_ratio)
            img = cv2.resize(img, (new_width, new_height))
            
            img_pil = ImageTk.PhotoImage(image=Image.fromarray(img))
            img_label.configure(image=img_pil)
            img_label.image = img_pil  # Keep a reference to prevent garbage collection

    root = tk.Tk()
    root.title("Vehicle Segmentation Application")
    root.geometry('1200x800')  # Adjust the size as needed for your screen

    # Load the image
    image_path = "data/test_images/autopalya-2-1024x576.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return
    
    # Variables for controls
    noise_type = tk.StringVar(value='gaussian')
    noise_intensity = tk.DoubleVar(value=0.1)
    preprocess_threshold = tk.DoubleVar(value=2)
    contrast_clip_limit = tk.DoubleVar(value=3.0)
    segmentation_area_ratio = tk.DoubleVar(value=0.005)
    min_aspect_ratio = tk.DoubleVar(value=0.2)
    max_aspect_ratio = tk.DoubleVar(value=5.0)
    min_solidity = tk.DoubleVar(value=0.5)
    postprocess_kernel_size = tk.IntVar(value=7)

    # Image display frames
    image_frame_top = Frame(root)
    image_frame_top.grid(row=0, column=0, columnspan=3, sticky='nsew', padx=5, pady=5)
    image_frame_bottom = Frame(root)
    image_frame_bottom.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)

    # Control panel
    control_frame = Frame(root)
    control_frame.grid(row=1, column=2, sticky='nsew', padx=5, pady=5)

    # Create image labels for displaying images
    image_labels = [Label(image_frame_top) for _ in range(3)] + [Label(image_frame_bottom) for _ in range(2)]
    for idx, img_label in enumerate(image_labels):
        if idx < 3:
            img_label.pack(side='left', expand=True, fill='both', padx=5, pady=5)
        else:
            img_label.pack(side='left', expand=True, fill='both', padx=5, pady=5)

    # Add control widgets to the control_frame
    Label(control_frame, text="Noise Type").pack()
    OptionMenu(control_frame, noise_type, 'gaussian', 'salt_pepper', command=update_processing).pack()
    control_vars = [
        ("Noise Intensity", Scale, noise_intensity, {'from_': 0, 'to': 1, 'resolution': 0.01}),
        ("Preprocessing Threshold", Scale, preprocess_threshold, {'from_': 0, 'to': 5, 'resolution': 0.1}),
        ("Contrast Clip Limit", Scale, contrast_clip_limit, {'from_': 1.0, 'to': 10.0, 'resolution': 0.1}),
        ("Segmentation Area Ratio", Scale, segmentation_area_ratio, {'from_': 0.001, 'to': 0.02, 'resolution': 0.001}),
        ("Minimum Aspect Ratio", Scale, min_aspect_ratio, {'from_': 0.1, 'to': 1.0, 'resolution': 0.1}),
        ("Maximum Aspect Ratio", Scale, max_aspect_ratio, {'from_': 1.0, 'to': 10.0, 'resolution': 0.1}),
        ("Minimum Solidity", Scale, min_solidity, {'from_': 0.1, 'to': 1.0, 'resolution': 0.1}),
        ("Postprocessing Kernel Size", Scale, postprocess_kernel_size, {'from_': 3, 'to': 15, 'resolution': 2})
    ]

    for label_text, widget, var, kwargs in control_vars:
        Label(control_frame, text=label_text).pack()
        widget(control_frame, variable=var, orient='horizontal', command=update_processing, **kwargs).pack()
    
    update_processing()

    root.mainloop()

if __name__ == '__main__':
    main()