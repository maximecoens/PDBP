# open images view
import cv2

image = cv2.imread('src\/testdata\juist90p2.jpg')

image = cv2.resize(image, (318, 691), interpolation=cv2.INTER_LINEAR)
cv2.imshow("Open CV Image", image)
cv2.waitKey(0)