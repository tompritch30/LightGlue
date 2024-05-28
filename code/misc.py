
# m_kpts0 = matchedkeypoints image 0
# print("Shape:", m_kpts0.shape)
# print("Data type:", m_kpts0.dtype)
# print("First elements:", m_kpts0)
# print("Mean:", m_kpts0.mean().item())
# print("Standard deviation:", m_kpts0.std().item())
# print("Min value:", m_kpts0.min().item())
# print("Max value:", m_kpts0.max().item())


def plot_matches_with_colored_keypoints(image0, image1, keypoints1, keypoints2, matches, error_dict_image0, error_dict_image1):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot each image
    for ax, img, keypoints, error_dict in zip(axes, [image0, image1], [keypoints1, keypoints2], [error_dict_image0, error_dict_image1]):
        if img.ndim == 3 and img.shape[0] == 3:
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            img = np.transpose(img, (1, 2, 0))
        ax.imshow(img)
        ax.axis('off')

        # Plot keypoints with colors based on error
        for x, y in keypoints:
            keypoint_tuple = (x, y)
            if keypoint_tuple in error_dict:
                error = error_dict[keypoint_tuple]
                normalized_error = (error - min(error_dict.values())) / (max(error_dict.values()) - min(error_dict.values()))
                color = plt.get_cmap('coolwarm')(normalized_error)
            else:
                color = 'lime'  # Default color for keypoints not in error_dict

            ax.scatter(x, y, c=[color], s=10, marker='o')  # Scatter plot for keypoints


    # Filter out invalid matches
    valid_matches = [(idx1, idx2) for idx1, idx2 in matches01['matches'] if
                     idx1 < len(keypoints1) and idx2 < len(keypoints2)]

    # Draw lines for matches in lime color
    for (idx1, idx2) in valid_matches:
        # print("there is a valid match!!", idx1, idx2)
        x0, y0 = keypoints1[idx1]
        x1, y1 = keypoints2[idx2]
        print("coordinates, ", x0, y0, x1, y1)
        # x1 += image0.shape[2] if image0.ndim == 3 else image0.shape[1]
        # Swap xyA and xyB ?
        con = ConnectionPatch(xyA=(x1, y1), xyB=(x0 + image0.shape[1], y0), coordsA="data", coordsB="data",
                              axesA=axes[1], axesB=axes[0], color='lime')  # Lime lines
        axes[1].add_artist(con)

    plt.tight_layout()
    plt.show()

plot_matches_with_colored_keypoints(image0, image1, m_kpts0, m_kpts1, matches01['matches'], error_dict_image0, error_dict_image1)

# def plot_matches_with_colors(image0, image1, keypoints1, keypoints2, matches, error_dict):
#     """Plots two images side-by-side, connecting matched keypoints with lines colored based on epipolar error."""
#
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
#     # Plot each image
#     for ax, img in zip(axes, [image0, image1]):
#         if img.ndim == 3 and img.shape[0] == 3:
#             if isinstance(img, torch.Tensor):
#                 img = img.numpy()
#             img = np.transpose(img, (1, 2, 0))  # Transpose for plt.imshow
#         ax.imshow(img)
#         ax.axis('off')
#
#     # Draw lines for valid matches, colored by epipolar error
#     for (idx1, idx2) in matches:
#         # if 0 <= idx1 < len(keypoints1) and 0 <= idx2 < len(keypoints2):
#             x0, y0 = keypoints1[idx1]
#             x1, y1 = keypoints2[idx2]
#             x1 += image0.shape[2] if image0.ndim == 3 else image0.shape[1]
#
#             keypoint_tuple = (x0, y0)  # Use as dictionary key
#             print(keypoint_tuple)
#
#             if keypoint_tuple in error_dict:
#                 error = error_dict[keypoint_tuple]
#                 normalized_error = (error - min(error_dict.values())) / (
#                             max(error_dict.values()) - min(error_dict.values()))
#                 color = plt.get_cmap('coolwarm')(normalized_error)  # Adjust colormap as needed
#             else:
#                 color = 'gray'  # Default color for keypoints not in error_dict
#
#             con = ConnectionPatch(xyA=(x1, y1), xyB=(x0, y0), coordsA="data", coordsB="data",
#                                   axesA=axes[1], axesB=axes[0], color=color)
#             axes[1].add_artist(con)
#
#     plt.show()
#
# # Generate colors based on epipolar error
# colors = plotting.map_errors_to_colors(epipolar_error)
#
# # Example usage
# plot_matches_with_colors(image0, image1, m_kpts0, m_kpts1, matches01['matches'], error_dict)

# colors = plotting.map_errors_to_colors(epipolar_error)
#
# # fig, axes = plotting.plot_images_with_matches([image0, image1], m_kpts0, m_kpts1, matches01["matches"], colors)
# # plt.show()
#
# ### Plotting epipolar graph ###
# # # Map errors to colors
# print("NOW trying to plot the epipolar graph \n\n")
# colors = plotting.map_errors_to_colors(epipolar_error)
# print(colors)
#
# axes = viz2d.plot_images([image0, image1])
# viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
# viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)


# # # Plot images
# axes = plotting.plot_images([image0, image1])
# print("Axes:", axes)
#
# # Calculate the offset for the second image
# print(image0.shape)
# image_width = image0.shape[2] # if image0.ndim == 3 else image0.shape[0]
#
# # # Plot matches with color based on epipolar error
# for i, (pt1, pt2, color) in enumerate(zip(m_kpts0, m_kpts1, colors)):
#     x0, y0 = pt1
#     x1, y1 = pt2
#     # x1 += image_width  # Offset x-coordinate of the second image
#     axes[0].plot([x0, x1], [y0, y1], color=color, lw=1.0)  # thicker lines
#
# viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
#
# kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
# viz2d.plot_images([image0, image1])
# viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
#
# # Show plot
# plt.show()

# From readme
# # # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
# # extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
# # matcher = LightGlue(features='disk').eval().cuda()  # load the matcher
#
# # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
# image0 = load_image(images / "000011_left.png").cuda()
# image1 = load_image(images / "000012_left.png").cuda()
#
# # extract local features
# feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
# feats1 = extractor.extract(image1)

# # match the features
# matches01 = matcher({'image0': feats0, 'image1': feats1})
# feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
# matches = matches01['matches']  # indices with shape (K,2)
# points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
# points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
