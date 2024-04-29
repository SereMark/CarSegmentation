import argparse
from utils import image_io, visualization
from processing import preprocess, noise, segmentation, postprocess

def process_images(image_path, noise_type, noise_intensity):
    original_img = image_io.load_image(image_path)
    noisy_img = noise.add_noise(original_img, noise_type, noise_intensity)
    preprocessed_img = preprocess.adaptive_preprocess(noisy_img)
    segmented_img = segmentation.segment_vehicles(original_img, preprocessed_img)
    postprocessed_img = postprocess.postprocess_segmentation(segmented_img)
    return [original_img, noisy_img, preprocessed_img, segmented_img, postprocessed_img]

def main(args):
    images = process_images(args.image_path, args.noise_type, args.noise_intensity)
    visualization.display_interactive_results(['Original', 'Noisy', 'Preprocessed', 'Segmented', 'Postprocessed'], images, lambda nt, ni: process_images(args.image_path, nt, ni))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vehicle Segmentation Application')
    parser.add_argument('--image-path', type=str, default='data/test_images/autopalya-2-1024x576.jpg', help='Path to the input image for segmentation')
    parser.add_argument('--add-noise', action='store_true', help='Flag to add noise to the image')
    parser.add_argument('--noise-type', choices=['gaussian', 'salt_pepper'], default='gaussian', help='Type of noise to add')
    parser.add_argument('--noise_intensity', type=float, default=0.1, help='Intensity of the noise')
    args = parser.parse_args()
    main(args)