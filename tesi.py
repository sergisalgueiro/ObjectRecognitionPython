# Package imports
import numpy as np
import argparse
import cv2
import funcutils as util

# Import image 
image = cv2.imread(r'C:\Users\salgueir\Pictures/P1291146.JPG')
# Resize image
imres = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
# cv2.imshow('img',imres)
# cv2.waitKey(0)
# cv2.destroyWindow('img')
print('Original image ' + str(image.shape) + ' -> Resized by 1/5 ->  ' + 'Resized image ' + str(imres.shape))

# Convert Image to HSV colorspace
imreshsv = cv2.cvtColor(imres, cv2.COLOR_BGR2HSV)
normimreshsv = util.normalizeHSV(imreshsv)
# cv2.imshow('HSVimg',imreshsv)
# cv2.waitKey(0)
# cv2.destroyWindow('HSVimg')

# Create 4D histogram
H = np.array(normimreshsv[:, :, 0]).ravel()
S = np.array(normimreshsv[:, :, 1]).ravel()
V = np.array(normimreshsv[:, :, 2]).ravel()

h3d = util.hist3d(normimreshsv, 25)
# Seeding
peaks3d = util.findPeak4D(h3d, 1)
print('End of program')
