import numpy as np
import cv2

def normalizeHSV(rawHSV):
	# For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
	normalized = np.empty(rawHSV.shape)
	normalized[:,:,0] = rawHSV[:,:,0]/179.0
	normalized[:,:,1] = rawHSV[:,:,1]/255.0
	normalized[:,:,2] = rawHSV[:,:,2]/255.0
	return normalized

def hist3d(img, intervals):
	space = np.linspace(0,1,intervals+1)
	h3d = np.zeros((intervals,intervals,intervals))
	for n in xrange(img.shape[0]):
		for m in xrange(img.shape[1]):
			tmpVal1 = img[n,m,0]
			tmpVal2 = img[n,m,1]
			tmpVal3 = img[n,m,2]
			idxVal1 = 0
			idxVal2 = 0
			idxVal3 = 0
			for i in xrange(intervals):
				if i == intervals:
					if ((tmpVal1 >= space[i]) and (tmpVal1 <= space[i+1])):
						idxVal1 = i
						break
				else:
					if ((tmpVal1 >= space[i]) and (tmpVal1 < space[i+1])):
						idxVal1 = i
						break
			for i in xrange(intervals):
				if i == intervals:
					if ((tmpVal2 >= space[i]) and (tmpVal2 <= space[i+1])):
						idxVal2 = i
						break
				else:
					if ((tmpVal2 >= space[i]) and (tmpVal2 < space[i+1])):
						idxVal2 = i
						break
			for i in xrange(intervals):
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
    for i in xrange(width,h3d.shape[0]-width):
        for j in xrange(width,h3d.shape[1]-width):
            for k in xrange(width,h3d.shape[2]-width):
                pattern = h3d[i-width:i+width+1,j-width:j+width+1,k-width:k+width+1]
                val = 6*pattern[1,1,1] - pattern[1,1,0] - pattern[1,1,2] - pattern[0,1,1] - pattern[1,0,1] -pattern[1,2,1] - pattern[2,1,1]
                # Afegir que el valor maxim sigui al pla mitj
                if val>0:
                    peaks = np.vstack((peaks,[i,j,k,val]))
    return peaks

def peakMask(peaks3d, clustThresh):
	count = 0
	peaks = peaks3d[peaks3d[:,3].argsort()[::-1]]
	print 'peaks'
	print peaks
	while True:
		if count == 0:
			peakMask = np.ones(peaks3d.shape[0], dtype = bool)
		# else:
			# peakMask = peaks3d[:, 1] > (coords(1, 1) + 1) | peaks3d[:, 1] < (coords(1, 1) - 1) & ...
   #                 peaks3d[:, 2] > (coords(1, 2) + 1) | peaks3d[:, 2] < (coords(1, 2) - 1) & ...
   #                 peaks3d[:, 3] > (coords(1, 3) + 1) | peaks3d[:, 3] < (coords(1, 3) - 1);
		newpeaks = peaks3d[peakMask, :]
		clustVal = newpeaks[:,3].max()
		if clustVal < clustThresh:
			break
		coords = newpeaks[0,:3]#si el primer es el mes alt...
		print 'coords'
		print coords
		clustCoords = np.vstack((clustCoords,coords))
		oldMask = peakMask
		oldClustVal = clustVal
		count += 1
		break
	return clustCoords

