from utils.noise import add_noise
from .preprocess import adaptive_preprocess
from .segmentation import segment_vehicles
from .postprocess import postprocess_segmentation

def process_image_pipeline(image, noise_type='gaussian', noise_intensity=0.1):
    noisy_img = add_noise(image, noise_type, noise_intensity)
    preprocessed_img = adaptive_preprocess(noisy_img.copy())
    segmented_img = segment_vehicles(image, preprocessed_img, min_area_ratio=0.005, aspect_ratio_range=(0.2, 5.0), min_solidity=0.5)
    postprocessed_img = postprocess_segmentation(segmented_img.copy())
    return [image, noisy_img, preprocessed_img, segmented_img, postprocessed_img]