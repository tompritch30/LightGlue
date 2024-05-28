
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

# # Plotting normal graph
# axes = viz2d.plot_images([image0, image1])
# viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=1)
# viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
#
# pts0 = np.array([list(pt) for pt in error_dict_image0.keys()])
# pts1 = np.array([list(pt) for pt in error_dict_image1.keys()])
#
# errors = list(error_dict_image0.values())


# Feats 0 is a dict_keys(['keypoints', 'keypoint_scores', 'descriptors', 'image_size'])
# keypoints is tensor([[ [287.3125, 304.8125], ... ]]) i.e. is a coordinate for each point
# score: tensor([[0.5934, 0.5182, 0.5137, 0.4981]]) i.e. confidence score 0-1 for all
# descriptor : tensor([[[ 0.0531,w/ 256 elements], ]]) is a 2D list : [[ [256 elements], [256 elements] etc ... ]]
# image_size, Value: tensor([[640., 480.]]) - in px

# print(type(feats0))
# print(feats0.keys())
# for key, value in feats0.items():
#     if isinstance(value, (list, tuple)):
#         print(f"Key: {key}, Sample Value: {value[:5]}")  # Print first 5 elements if it's a list or tuple
#     elif isinstance(value, np.ndarray):  # Assuming you are using NumPy arrays
#         print(f"Key: {key}, Shape: {value.shape}, Data Type: {value.dtype}")
#     else:
#         print(f"Key: {key}, Value: {value}")
# print(feats0, "\n\n", feats0)

# print("\n\n\nhere are the errors\n:", pts0, pts1, errors, sep="\n")
# pts0 = np.array([[137.9375, 64.1875], [191.6875, 59.1875], [345.4375, 228.5625], [148.5625, 280.4375],
#         [159.1875, 47.9375], [576.6875, 115.4375], [520.4375, 396.6875], [175.4375, 37.9375],
#         [456.0625, 297.9375], [264.1875, 106.0625], [538.5625, 439.8125], [301.0625, 40.4375],
#         [251.0625, 62.9375], [577.9375, 87.3125], [347.9375, 11.6875], [247.9375, 79.8125],
#         [439.8125, 7.9375], [457.9375, 112.9375]])
#
# pts1 = np.array([[99.8125, 69.8125], [162.9375, 59.1875], [311.6875, 251.6875], [119.8125, 280.4375],
#         [139.1875, 141.0625], [476.0625, 118.5625], [457.3125, 377.3125], [145.4375, 42.9375],
#         [354.8125, 288.5625], [361.0625, 134.1875], [416.6875, 364.8125], [229.1875, 41.0625],
#         [220.4375, 59.1875], [476.0625, 76.0625], [296.0625, 22.3125], [231.0625, 63.5625],
#         [493.5625, 16.0625], [449.8125, 184.1875]])
#
# errors = [0.002907528241603695, 0.0028924842188165186, 0.002880773161212839, 0.0028800448095319545,
#           0.0028697429621220316, 0.002983036980895951, 0.002913153503132172, 0.00289568311886416,
#           0.0029783167523154777, 0.0026981768636036556, 0.0030059265704460757, 0.0029546556814259994,
#           0.0028933542788950996, 0.0029885241513693834, 0.002923248583335889, 0.0028733440246676307,
#           0.0027632819479406325, 0.002839756308815569]


# viz2d.plot_colored_matches(pts0, pts1, errors)
# plt.show()
#
#
# axis = viz2d.plot_images([image0, image1])
# viz2d.plot_colored_matches(pts0, pts1, errors)
# # viz2d.plot_matches(pts0, pts1, color="red", lw=1)
# viz2d.add_text(0, f'Testing points from dict', fs=20)

# kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
# viz2d.plot_images([image0, image1])
# viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
# plt.show()

# print(len(pts0), len(pts1), len(errors))

####OLD EPIPOLAR #####

## Epipolar error
# https://github.com/johannes-graeter/momo
# Compute the Essential matrix (as an example)
# E, _ = cv2.findEssentialMat(m_kpts0, m_kpts1, poses.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
# # Compute epipolar errors
# epipolar_error = poses.compute_epipolar_error(E, m_kpts0, m_kpts1)
# print("Normal Epipolar Error:\n", epipolar_error)
# # Construct transformation matrix T_0to1
# T_0to1 = np.linalg.inv(pose_matrix2) @ pose_matrix1
#
# print(kpts0, kpts1, T_0to1, poses.K)
#
# # Use the SuperGlue method to compute epipolar error
# epipolar_error = poses.compute_superglue_epipolar_error(kpts0, kpts1, T_0to1, poses.K, poses.K)
# print("My SuperGlue Epipolar Error:\n", epipolar_error)


# Convert pts0 and pts1 to numpy arrays if they are not already
# pts0 = np.array(pts0)
# pts1 = np.array(pts1)

# # Normalize error values to [0, 1] for colormap
# threshold = 4000  # Adjust this threshold based on your data
# normalized_errors = [min(e / threshold, 1.0) for e in errors]
# colors = [viz2d.cm_RdGn(e) for e in normalized_errors]
