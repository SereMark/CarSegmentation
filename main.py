import argparse
from utils.image_io import load_image
from processing.process_image_pipeline import process_image_pipeline
from visualization.display import display_interactive_results

def parse_arguments():
    parser = argparse.ArgumentParser(description='Vehicle Segmentation Application')
    parser.add_argument('--image-path', type=str, default='data/test_images/autopalya-2-1024x576.jpg',
                        help='Path to the input image for segmentation. Default is "data/test_images/autopalya-2-1024x576.jpg"')
    parser.add_argument('--noise-type', choices=['gaussian', 'salt_pepper'], default='gaussian',
                        help='Type of noise to add')
    parser.add_argument('--noise-intensity', type=float, default=0.1, help='Intensity of the noise')
    return parser.parse_args()

def main():
    args = parse_arguments()
    image = load_image(args.image_path)
    results = process_image_pipeline(image.copy(), args.noise_type, args.noise_intensity)
    titles = ['Original', 'Noisy', 'Preprocessed', 'Segmented', 'Postprocessed']
    display_interactive_results(titles, results)

if __name__ == '__main__':
    main()