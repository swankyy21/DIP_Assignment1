import cv2
import numpy as np

def isOriginal(image1, image2):
      grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
      grayImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

      score1 = cv2.Laplacian(grayImage1, cv2.CV_64F)
      varianceScore1 = score1.var()
      score2 = cv2.Laplacian(grayImage2, cv2.CV_64F)
      varianceScore2 = score2.var()

      print(varianceScore1)
      print(varianceScore2)

      if varianceScore1>varianceScore2:
            return "Original Image", "Blurred Image"
      else:
            return "Blurred Image", "Original Image"

image1 = cv2.imread('./data/fig5.jpg')
image2 = cv2.imread('./data/fig5_blur.jpg')
original, blurred = isOriginal(image1, image2)

cv2.imshow(original, image1)
cv2.imshow(blurred, image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

