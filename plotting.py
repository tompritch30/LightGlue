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