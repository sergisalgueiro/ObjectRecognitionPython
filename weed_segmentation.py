# Package imports
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import funcutils as util


def main(color_space):
    # Read image
    image = cv2.imread(r'C:\Users\salgueir\Pictures/zucchini.jpg')
    # Resize image
    # image = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    # Convert Image to HSV colorspace
    im_channels = cv2.cvtColor(image, color_space)

    # Plot pixel distribution
    # util.scatter_image(im_channels)
    # Apply histogram equalization for value
    # util.hist_eq_hsv(im_channels)
    # Applu Gaussian blur
    im_channels = cv2.GaussianBlur(im_channels, (5, 5), 0)
    # KMEANS clustering
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
    plt.show()


if __name__ == "__main__":
    main(cv2.COLOR_BGR2HSV)
    main(cv2.COLOR_BGR2RGB)
