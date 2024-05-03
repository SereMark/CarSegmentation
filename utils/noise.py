import numpy as np

def add_noise(image, noise_type='gaussian', intensity=0.1):
    noise_functions = {
        'gaussian': lambda img: add_gaussian_noise(img, intensity),
        'salt_pepper': lambda img: add_salt_pepper_noise(img, intensity)
    }
    noise_func = noise_functions.get(noise_type)
    return noise_func(image)

def add_gaussian_noise(image, intensity):
    row, col, ch = image.shape
    mean = 0
    sigma = intensity * 255
    gauss_noise = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
    noisy_image = np.clip(image + gauss_noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_pepper_noise(image, intensity):
    out_image = image.copy()
    row, col, ch = out_image.shape
    s_vs_p = 0.5
    total_pixels = row * col * ch
    amount = int(total_pixels * intensity)
    num_salt = int(amount * s_vs_p)
    num_pepper = amount - num_salt

    salt_coords = [np.random.randint(0, i, num_salt) for i in (row, col, ch)]
    out_image[tuple(salt_coords)] = 255

    pepper_coords = [np.random.randint(0, i, num_pepper) for i in (row, col, ch)]
    out_image[tuple(pepper_coords)] = 0

    return out_image