import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import funcutils as util

sensitivity = 20
lower_green = np.array([60 - sensitivity, 100, 30])
upper_green = np.array([60 + sensitivity, 255, 255])


def load_hsv_image(img_path, image_training_size, plot=False):
    # Read image
    # image = cv2.imread(r'C:\Users\salgueir\Pictures/zucchini.jpg')
    image = cv2.imread(img_path)
    # Resize image
    image = cv2.resize(image, image_training_size)
    # Convert Image to HSV color space
    im_channels = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Apply Gaussian blur
    im_channels = cv2.GaussianBlur(im_channels, (5, 5), 0)
    if plot:
        # Plot pixel distribution
        util.scatter_image(im_channels)
        # Apply histogram equalization for value
        util.hist_eq_hsv(im_channels)
    # static_clustering(im_channels)
    # adaptive_clustering(im_channels, 25, 5)
    return im_channels


def adaptive_clustering(hsv_img, num_bins, width, plot=False):
    im_norm = util.normalizeHSV(hsv_img)
    h3d = util.hist3d(im_norm, num_bins)
    p4d = util.findPeak4D(h3d, width)
    peaks_center = util.peakMask(p4d, 40)
    # K-MEANS clustering
    k = len(peaks_center)
    assert k != 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Sort every channel in a single column
    z = hsv_img.reshape((-1, 3))
    # convert to np.float32
    z = np.float32(z)
    ret, label, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(hsv_img.shape)
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
        ax2.imshow(res2)
        plt.show()
    # Get mean HSV for every Cluster
    cluster_mean = np.ndarray((k, 3))
    for i in range(k):
        cluster_mean[i] = z[label.flatten() == i].mean(axis=0)
    # Extract green clusters
    green_clus = util.in_range(cluster_mean, lower_green, upper_green)
    idx = np.nonzero(green_clus)
    # Make binary image with all green clusters
    mask = np.isin(label.flatten(), idx)
    bin_im = mask.reshape(hsv_img.shape[:2])
    if plot:
        plt.imshow(bin_im)
        plt.show()
    return bin_im


def static_clustering(hsv_img, plot=False):
    # K-MEANS clustering
    # Sort every channel in a single column
    z = hsv_img.reshape((-1, 3))
    # convert to np.float32
    z = np.float32(z)
    # define criteria, number of clusters(K) and apply k-means()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2
    ret, label, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(hsv_img.shape)
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
        ax2.imshow(res2)
        plt.show()
    # Create binary image
    img_binary = label.reshape(hsv_img.shape[:2])
    return img_binary


def morphological_operations(img_binary, hsv_img, plot=False):
    # MORPHOLOGICAL OPERATIONS
    img_binary = np.uint8(img_binary)
    # Open/close operation
    kernel_size = 3
    img_oc = util.open_close(img_binary, kernel_size)
    # img_co = util.close_open(img_oc, kernel_size*2)
    if plot:
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        plt.imshow(img_oc)
        plt.show()
    # Check for connected components ang get bounding box
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_oc)
    if len(stats) < 2:
        return None
    area_value_kernel = np.sort(stats[:, cv2.CC_STAT_AREA])[-2]
    bounding_box = stats[np.where(stats[:, cv2.CC_STAT_AREA] == area_value_kernel), :].flatten()
    # Plotting
    if plot:
        plt_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        cv2.rectangle(plt_img,
                      tuple(bounding_box[[cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP]]),
                      tuple(bounding_box[[cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP]] + bounding_box[[cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]),
                      1, 1)
        cv2.imshow('Img', plt_img)
        cv2.waitKey(0)
        cv2.destroyWindow('Img')
    return bounding_box


def get_green_bounding_box(path_to_img, size=(224, 224), mode='adaptive'):
    im_hsv = load_hsv_image(path_to_img, size)
    if mode == 'adaptive':
        bin_im = adaptive_clustering(im_hsv, 25, 5, False)
    else:
        raise LookupError("Mode not supported")
    bbox = morphological_operations(bin_im, im_hsv, False)
    return bbox


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('PATH')
    args = parse.parse_args()
    get_green_bounding_box(args.PATH)
