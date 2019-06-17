# Package imports
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import funcutils as util
from time import sleep


def main(color_space):
    # Read image
    image = cv2.imread(r'C:\Users\salgueir\Pictures/zucchini.jpg')
    # Resize image
    # image = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    # Convert Image to HSV color space
    im_channels = cv2.cvtColor(image, color_space)

    # Plot pixel distribution
    # util.scatter_image(im_channels)
    # Apply histogram equalization for value
    # util.hist_eq_hsv(im_channels)
    # Apply Gaussian blur
    im_channels = cv2.GaussianBlur(im_channels, (5, 5), 0)

    # K-MEANS clustering
    # Sort every channel in a single column
    z = im_channels.reshape((-1, 3))
    # convert to np.float32
    z = np.float32(z)
    # define criteria, number of clusters(K) and apply k-means()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2
    ret, label, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(im_channels.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax2.imshow(res2)
    # plt.show(block=False)
    plt.show()

    # MORPHOLOGICAL OPERATIONS
    # Create binary image
    img_binary = label.reshape(im_channels.shape[:2])
    img_binary = np.uint8(img_binary)
    # Open/close operation
    kernel_size = 50
    img_oc = util.open_close(img_binary, kernel_size)
    img_co = util.close_open(img_oc, kernel_size*2)
    # Check for connected components ang get bounding box
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_co)
    area_value_kernel = np.sort(stats[:, cv2.CC_STAT_AREA])[-2]
    bounding_box = stats[np.where(stats[:, cv2.CC_STAT_AREA] == area_value_kernel), :].flatten()
    # Plotting
    plt_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(plt_img.shape)
    cv2.rectangle(plt_img,
                  tuple(bounding_box[[cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP]]),
                  tuple(bounding_box[[cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP]] + bounding_box[[cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]),
                  1, 1)
    cv2.imshow('Img', plt_img)
    cv2.waitKey(0)
    cv2.destroyWindow('Img')
	#lower_green = np.array([65,60,60])
	#upper_green = np.array([80,255,255])
	#mask = cv2.inRange(hsv, lower_green, upper_green)


if __name__ == "__main__":
    main(cv2.COLOR_BGR2HSV)
    # main(cv2.COLOR_BGR2RGB)
