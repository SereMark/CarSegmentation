import matplotlib.pyplot as plt
import cv2

def display_interactive_results(titles, images):
    """Display images with titles in a matplotlib plot."""
    plt.figure(figsize=(10, 8))
    for i, (title, img) in enumerate(zip(titles, images), 1):
        plt.subplot(2, 3, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.show()