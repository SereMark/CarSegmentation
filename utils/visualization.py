import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

def display_interactive_results(titles, images, process_function):
    fig, axs = plt.subplots(2, 3, figsize=(30, 20), dpi=100)
    axs = axs.flatten()

    imgs = []
    for ax, img, title in zip(axs[:-1], images, titles):
        imgs.append(ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        ax.set_title(title, fontdict={'fontsize': 18, 'fontweight': 'bold'})
        ax.axis('off')

    axs[-1].axis('off')

    axcolor = '#caf0f8'
    slider_color = '#023e8a'
    ax_noise_type = plt.axes([0.775, 0.1, 0.15, 0.15], facecolor=axcolor, transform=fig.transFigure)
    ax_noise_intensity = plt.axes([0.775, 0.3, 0.15, 0.03], facecolor=axcolor, transform=fig.transFigure)

    noise_type_button = RadioButtons(ax_noise_type, ('Gaussian', 'Salt & Pepper'), active=0, activecolor=slider_color)
    noise_intensity_slider = Slider(ax_noise_intensity, 'Noise Intensity', 0.0, 1.0, valinit=0.1, color=slider_color)

    def update(val):
        noise_type = 'gaussian' if noise_type_button.value_selected == 'Gaussian' else 'salt_pepper'
        noise_intensity = noise_intensity_slider.val
        updated_images = process_function(noise_type, noise_intensity)
        for img, upd_img in zip(imgs, updated_images):
            img.set_data(cv2.cvtColor(upd_img, cv2.COLOR_BGR2RGB))
        fig.canvas.draw_idle()

    noise_type_button.on_clicked(update)
    noise_intensity_slider.on_changed(update)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.3)
    plt.show()