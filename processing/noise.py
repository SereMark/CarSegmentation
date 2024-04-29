import cv2
import numpy as np

def add_gaussian_noise(image, intensity=0.1):
    """ Add Gaussian noise to the image. """
    row, col, ch = image.shape
    mean = 0
    sigma = intensity * 255
    gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch).astype(np.float32)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # Ensure pixel values stay within valid range
    return noisy_image

def add_salt_pepper_noise(image, intensity=0.1):
    """ Add salt-and-pepper noise to the image. """
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = intensity * 0.5
    noisy_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[tuple(coords)] = 255
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p)).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[tuple(coords)] = 0
    return noisy_image

def add_noise(image, noise_type='gaussian', intensity=0.1):
    """ Apply selected noise type to the image. """
    if noise_type == 'gaussian':
        return add_gaussian_noise(image, intensity)
    elif noise_type == 'salt_pepper':
        return add_salt_pepper_noise(image, intensity)