import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D


def normalizeHSV(rawHSV):
	# For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
	normalized = np.empty(rawHSV.shape)
	normalized[:, :, 0] = rawHSV[:, :, 0]/179.0
	normalized[:, :, 1] = rawHSV[:, :, 1]/255.0
	normalized[:, :, 2] = rawHSV[:, :, 2]/255.0
	return normalized


def get_autoscaled_colors(image):
	pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
	norm = colors.Normalize(vmin=-1., vmax=1.)
	norm.autoscale(pixel_colors)
	pixel_colors = norm(pixel_colors).tolist()
	return pixel_colors


def hist_eq_hsv(image):
	_, _, value = cv2.split(image)
	value_hist = np.ndarray(shape=value.shape, dtype=value.dtype)
	image[:, :, 2] = value_hist
	return image


def scatter_image(image):
	# Get pixel colors for the scatter plot
	pixel_colors = get_autoscaled_colors(image)
	# split image into 3 channels
	chan1, chan2, chan3 = cv2.split(image)
	fig = plt.figure()
	axis = fig.add_subplot(111, projection="3d")
	axis.scatter(chan1.flatten(), chan2.flatten(), chan3.flatten(), facecolors=pixel_colors, marker=".")
	axis.set_xlabel("Hue")
	axis.set_ylabel("Saturation")
	axis.set_zlabel("Value")
	plt.show()
	return


def hist3d(img, intervals):
	space = np.linspace(0,1,intervals+1)
	h3d = np.zeros((intervals,intervals,intervals))
	for n in range(img.shape[0]):
		for m in range(img.shape[1]):
			tmpVal1 = img[n,m,0]
			tmpVal2 = img[n,m,1]
			tmpVal3 = img[n,m,2]
			idxVal1 = 0
			idxVal2 = 0
			idxVal3 = 0
			for i in range(intervals):
				if i == intervals:
					if ((tmpVal1 >= space[i]) and (tmpVal1 <= space[i+1])):
						idxVal1 = i
						break
				else:
					if ((tmpVal1 >= space[i]) and (tmpVal1 < space[i+1])):
						idxVal1 = i
						break
			for i in range(intervals):
				if i == intervals:
					if ((tmpVal2 >= space[i]) and (tmpVal2 <= space[i+1])):
						idxVal2 = i
						break
				else:
					if ((tmpVal2 >= space[i]) and (tmpVal2 < space[i+1])):
						idxVal2 = i
						break
			for i in range(intervals):
				if i == intervals:
					if ((tmpVal3 >= space[i]) and (tmpVal3 <= space[i+1])):
						idxVal3 = i
						break
				else:
					if ((tmpVal3 >= space[i]) and (tmpVal3 < space[i+1])):
						idxVal3 = i
						break
			h3d[idxVal1,idxVal2,idxVal3] = h3d[idxVal1,idxVal2,idxVal3] + 1
	return h3d


def findPeak4D(h3d, width):
    peaks = np.empty((0,4),int)
    for i in range(width,h3d.shape[0]-width):
        for j in range(width,h3d.shape[1]-width):
            for k in range(width,h3d.shape[2]-width):
                pattern = h3d[i-width:i+width+1,j-width:j+width+1,k-width:k+width+1]
                val = 6*pattern[1,1,1] - pattern[1,1,0] - pattern[1,1,2] - pattern[0,1,1] - pattern[1,0,1] -pattern[1,2,1] - pattern[2,1,1]
                # Afegir que el valor maxim sigui al pla mitj
                if val>0:
                    peaks = np.vstack((peaks,[i,j,k,val]))
    return peaks


def peakMask(peaks3d, clustThresh):
	count = 0
	peaks = peaks3d[peaks3d[:,3].argsort()[::-1]]
	print(peaks)
	print(peaks)
	while True:
		if count == 0:
			peakMask = np.ones(peaks3d.shape[0], dtype=bool)
		# else:
			# peakMask = peaks3d[:, 1] > (coords(1, 1) + 1) | peaks3d[:, 1] < (coords(1, 1) - 1) & ...
   #                 peaks3d[:, 2] > (coords(1, 2) + 1) | peaks3d[:, 2] < (coords(1, 2) - 1) & ...
   #                 peaks3d[:, 3] > (coords(1, 3) + 1) | peaks3d[:, 3] < (coords(1, 3) - 1);
		newpeaks = peaks3d[peakMask, :]
		clustVal = newpeaks[:,3].max()
		if clustVal < clustThresh:
			break
		coords = newpeaks[0, :3] #s i el primer es el mes alt...
		print('coords')
		print(coords)
		clustCoords = np.ndarray([])
		clustCoords = np.vstack((clustCoords, coords))
		oldMask = peakMask
		oldClustVal = clustVal
		count += 1
		break
	return clustCoords


def erosion(img, kernel_size, iterations):
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	erosion = cv2.erode(img, kernel, iterations=iterations)
	return erosion


def open_close(img, kernel_size):
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	img_oc = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	img_oc = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	return img_oc


def close_open(img, kernel_size):
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	img_co = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	img_co = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	return img_co
