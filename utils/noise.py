import numpy as np

def add_noise(image, noise_type='gaussian', intensity=0.1):
    if noise_type == 'gaussian':
        return add_gaussian_noise(image, intensity)
    elif noise_type == 'salt_pepper':
        return add_salt_pepper_noise(image, intensity)

def add_gaussian_noise(image, intensity=0.1):
    row, col, ch = image.shape
    mean = 0
    sigma = intensity * 255
    gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch).astype(np.float32)
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_pepper_noise(image, intensity=0.05):
    out_image = image.copy()
    row, col, ch = out_image.shape
    s_vs_p = 0.5
    amount = (row * col * ch) * intensity
    num_salt = int(amount * s_vs_p)
    num_pepper = int(amount - num_salt)

    # Add Salt noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in out_image.shape]
    out_image[tuple(coords)] = 255

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in out_image.shape]
    out_image[tuple(coords)] = 0

    return out_image