import cv2
import numpy as np


def calculateArea(pathOfImage):
    originalImage = cv2.imread(pathOfImage)
    cv2.imshow('Original Image', originalImage)
    hsvImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)


    #Blue Color Segment
    blueLower = np.array([110, 50, 50])
    blueUpper = np.array([130, 255, 255])
    mask1 = cv2.inRange(hsvImage, blueLower, blueUpper)
    detected1 = cv2.bitwise_and(originalImage, originalImage, mask=mask1)
    cnts1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    maxWidth1 = 0
    maxLength1 = 0
    for c in cnts1:
        x, y, w, h = cv2.boundingRect(c)
        maxWidth1 = max(maxWidth1, w)
        maxLength1 = max(maxLength1, h)
        cv2.drawContours(originalImage, [c], 0, (0, 255, 0), 2)
    print("Width of blue segment :", maxWidth1)
    print("Length of blue segment :", maxLength1)

    cv2.imshow('Blue Color Image', detected1)
    
    
    #Green Color Segment
    greenLower = np.array([50, 100, 100])
    greenUpper = np.array([70, 255, 255])
    mask2 = cv2.inRange(hsvImage, greenLower, greenUpper)
    detected2 = cv2.bitwise_and(originalImage, originalImage, mask=mask2)
    cnts2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    maxWidth2 = 0
    maxLength2 = 0
    for c in cnts2:
        x, y, w, h = cv2.boundingRect(c)
        maxWidth2 = max(maxWidth2, w)
        maxLength2 = max(maxLength2, h)
        cv2.drawContours(originalImage, [c], 0, (0, 255, 0), 2)
    print("\nWidth of green segment :", maxWidth2)
    print("Length of green segment :", maxLength2)
    
    cv2.imshow('Green Color Image', detected2)
    
    
    #Yellow Color Segment
    yellowLower = np.array([20, 100, 100])
    yellowUpper = np.array([30, 255, 255])
    mask3 = cv2.inRange(hsvImage, yellowLower, yellowUpper)
    detected3 = cv2.bitwise_and(originalImage, originalImage, mask=mask3)
    cnts3, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    maxWidth3 = 0
    maxLength3 = 0
    for c in cnts3:
        x, y, w, h = cv2.boundingRect(c)
        maxWidth3 = max(maxWidth3, w)
        maxLength3 = max(maxLength3, h)
        cv2.drawContours(originalImage, [c], 0, (0, 255, 0), 2)
    print("\nWidth of yellow segment :", maxWidth3)
    print("Length of yellow segment :", maxLength3)
    
    cv2.imshow('Yellow Color Image', detected3)
    
    
    #Light Blue Color Segment
    lightBlueLower = np.array([90, 100, 100])
    lightBlueUpper = np.array([110, 255, 255])
    mask4 = cv2.inRange(hsvImage, lightBlueLower, lightBlueUpper)
    detected4 = cv2.bitwise_and(originalImage, originalImage, mask=mask4)
    cnts4, _ = cv2.findContours(mask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    maxWidth4 = 0
    maxLength4 = 0
    for c in cnts4:
        x, y, w, h = cv2.boundingRect(c)
        maxWidth4 = max(maxWidth4, w)
        maxLength4 = max(maxLength4, h)
        cv2.drawContours(originalImage, [c], 0, (0, 255, 0), 2)
    print("\nWidth of light blue segment :", maxWidth4)
    print("Length of light blue segment :", maxLength4)
    
    cv2.imshow('Light Blue Color Image', detected4)
    
    
    #Brown Color Segment
    brownLower = np.array([10, 100, 100])
    brownUpper = np.array([20, 255, 255])
    mask5 = cv2.inRange(hsvImage, brownLower, brownUpper)
    detected5 = cv2.bitwise_and(originalImage, originalImage, mask=mask5)
    cnts5, _ = cv2.findContours(mask5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    maxWidth5 = 0
    maxLength5 = 0
    for c in cnts5:
        x, y, w, h = cv2.boundingRect(c)
        maxWidth5 = max(maxWidth5, w)
        maxLength5 = max(maxLength5, h)
        cv2.drawContours(originalImage, [c], 0, (0, 255, 0), 2)
    print("\nWidth of brown segment :", maxWidth5)
    print("Length of brown segment :", maxLength5)
    
    cv2.imshow('Brown Color Image', detected5)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Main Function
pathOfImage = './data/finger-bones.jpg'
calculateArea(pathOfImage)
