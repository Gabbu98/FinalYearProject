import cv2
import numpy as np

image = cv2.imread('img24.jpg')
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(image, -1, kernel)
#sharpened = cv2.filter2D(image, -1, kernel) # applying the sharpening kernel to the input image & displaying it.
cv2.imshow('Image Sharpening', sharpened)
cv2.imwrite('img24improved.jpg',sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()