import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np


def map_errors_to_colors(errors, colormap='RdYlGn_r'):
    """Map errors to colors using a specified colormap."""
    norm = Normalize(vmin=min(errors), vmax=max(errors))
    cmap = plt.get_cmap(colormap)
    return [cmap(norm(error)) for error in errors]


def plot_images(images):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, img in zip(axes, images):
        # Convert image to numpy array if not already
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Transpose image from (C, H, W) to (H, W, C) if needed
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        ax.imshow(img)
        ax.axis('off')
        ax.set_aspect('equal')
    return axes

    # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # for ax, img in zip(axes, images):
    #     # Transpose image from (C, H, W) to (H, W, C)
    #     if img.shape[0] == 3:
    #         img = img.transpose(1, 2, 0)
    #     ax.imshow(img)
    #     ax.axis('off')
    # return axes
    # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # for ax, img in zip(axes, images):
    #     ax.imshow(img)
    #     ax.axis('off')
    # return axes


def plot_images_with_matches(images, keypoints1, keypoints2, matches, colors):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, img in zip(axes, images):
        # Check if the image needs to be transposed
        if img.shape[0] == 3:  # Assuming the image is in (channels, height, width)
            img = img.transpose(1, 2, 0)  # Transpose to (height, width, channels)
        ax.imshow(img)
        ax.axis('off')

    # Assuming matches are indexed as (index in keypoints1, index in keypoints2)
    custom_plot_matches(images, axes, keypoints1, keypoints2, matches, colors, lw=1.0)  # Adjust line width as necessary
    return fig, axes

def custom_plot_matches(images, axes, keypoints1, keypoints2, matches, colors, lw=0.2):
    """Plot matches with individual colors on the provided axes."""
    for i, (idx1, idx2) in enumerate(matches):
        x0, y0 = keypoints1[idx1]
        x1, y1 = keypoints2[idx2]
        # Correct for image offset in x1 since it's plotted on the same axis
        x1 += images[0].shape[1] if images[0].ndim == 3 else images[0].shape[0]
        axes[0].plot([x0, x1], [y0, y1], color=colors[i], lw=lw)
