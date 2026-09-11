import cv2
import matplotlib.pyplot as plt
image = cv2.imread('images/1.png' , cv2.IMREAD_GRAYSCALE)
# cv2.imshow('image', image)
# cv2.waitKey(0)

plt.hist(image.ravel(),256,[0,256])
plt.show()