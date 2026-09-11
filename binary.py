import cv2

image = cv2.imread('images/1.png' , cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', image)
cv2.waitKey(0)
