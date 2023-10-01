import cv2
import numpy as np
import matplotlib.pyplot as plt


def gender_detector(pathOfImage):
    image = cv2.imread(pathOfImage)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filteredContours = [contour for contour in contours if 5000 < cv2.contourArea(contour) < 100000]

    cv2.drawContours(image, filteredContours, -1, (0, 255, 0), 3)
    length, width, channel = image.shape
    imageArea = length*width
    hairContour = cv2.contourArea(filteredContours[0])

    print("\nArea",imageArea)
    print("Hair Contour",hairContour)

    imaginaryThreshold = hairContour/imageArea
    # Given that hair area of girl compared to image area is 0.36 approx
    # And hair area of boy is 0.26, We can set threshold at 0.3-0.31 approx to differentiate
    if imaginaryThreshold > 0.31:
        print('\nImage is of a girl')
        cv2.imshow('Girl Image', image)
    else:
        print('\nImage is of a boy')
        cv2.imshow('Boy Image', image)


#Main Function
gender_detector('./data/fig3.jpg')
gender_detector('./data/fig4.jpg')
cv2.waitKey(0)