from matplotlib.pylab import norm
import tensorflow as tf
import numpy as np
import cv2

np.seterr(divide='ignore')

inf = np.Infinity
onf = np.arctanh(1)

print(5 > inf)
print(5 > onf)

