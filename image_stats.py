import glob

import cv2
import numpy as np

files = glob.glob("../../hpa/train/*_yellow.png")

sum = 0
for f in files:
	image = cv2.imread(f, cv2.IMREAD_GRAYSCALE) / 255.
	sum += np.sum(image)

mean = sum / len(files) / 512 / 512
print(mean)

sum = 0
for f in files:
	image = cv2.imread(f, cv2.IMREAD_GRAYSCALE) / 255.
	image2 = (image - mean)
	sum += (image2 * image2).sum()

stddev = np.sqrt(sum / len(files) / 512 / 512)
print(stddev)
