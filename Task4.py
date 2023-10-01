import cv2
import numpy as np


def calculateSegment(pathOfImage):
    originalImage = cv2.imread(pathOfImage)
    image1 = originalImage.copy()
    image2 = originalImage.copy()
    image3 = originalImage.copy()
    image4 = originalImage.copy()
    hsv_image = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

    yellowLower = np.array([20, 100, 100])
    yellowUpper = np.array([30, 255, 255])
    mask1 = cv2.inRange(hsv_image, yellowLower, yellowUpper)
    detected1 = cv2.bitwise_and(image1, image1, mask=mask1)
    cnts1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area1=0
    for c in cnts1:
        area1 += cv2.contourArea(c)
        M1 = cv2.moments(c)
        if M1["m00"]!=0:
            cx1 = int(M1["m10"] / M1["m00"])
            cy1 = int(M1["m01"] / M1["m00"])
            cv2.putText(originalImage, f"{area1:.2f}", (cx1 - 37, cy1), cv2.FONT_HERSHEY_TRIPLEX, 0.48, (0,0,0), 1)
        cv2.drawContours(image1,[c], 0, (0,0,0), 2)
    #cv2.imshow('Color Detection Result 1', detected1)
    print("Yellow :",area1)
    
    lightLower = np.array([0, 0, 220])
    lightUpper = np.array([180, 30, 250])
    mask2 = cv2.inRange(hsv_image, lightLower, lightUpper)
    detected2 = cv2.bitwise_and(image2, image2, mask=mask2)
    cnts2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area2=0
    for c in cnts2:
        area2 += cv2.contourArea(c)
        M2 = cv2.moments(c)
        if M2["m00"]!=0:
            cx2 = int(M2["m10"] / M2["m00"])
            cy2 = int(M2["m01"] / M2["m00"])
            cv2.putText(originalImage, f"{area2:.2f}", (cx2 - 35, cy2), cv2.FONT_HERSHEY_TRIPLEX, 0.48, (0,0,0), 1)
        cv2.drawContours(image2,[c], 0, (0,0,0), 2)
    #cv2.imshow('Color Detection Result 2', detected2)
    print("Light Gray :",area2)
    
    grayLower = np.array([0, 0, 200])
    grayUpper = np.array([220, 0, 250])
    mask3 = cv2.inRange(hsv_image, grayLower, grayUpper)
    detected3 = cv2.bitwise_and(image3, image3, mask=mask3)
    cnts3, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area3=0
    for c in cnts3:
        area3 += cv2.contourArea(c)
        M3 = cv2.moments(c)
        if M3["m00"]!=0:
            cx3 = int(M3["m10"] / M3["m00"])
            cy3 = int(M3["m01"] / M3["m00"])
            cv2.putText(originalImage, f"{area3:.2f}", (cx3 - 35, cy3), cv2.FONT_HERSHEY_TRIPLEX, 0.48, (0,0,0), 1)
        cv2.drawContours(image3,[c], 0, (0,0,0), 2)
    #cv2.imshow('Color Detection Result 3', detected3)
    print("Gray :",area3)
    
    darkLower = np.array([0, 0, 100])
    darkUpper = np.array([180, 30, 200])
    mask4 = cv2.inRange(hsv_image, darkLower, darkUpper)
    detected4 = cv2.bitwise_and(image4, image4, mask=mask4)
    cnts4, _ = cv2.findContours(mask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area4=0
    for c in cnts4:
        area4 += cv2.contourArea(c)
        M4 = cv2.moments(c)
        if M4["m00"]!=0:
            cx4 = int(M4["m10"] / M4["m00"])
            cy4 = int(M4["m01"] / M4["m00"])
            cv2.putText(originalImage, f"{area4:.2f}", (cx4 - 35, cy4), cv2.FONT_HERSHEY_TRIPLEX, 0.48, (0,0,0), 1)
        cv2.drawContours(image4,[c], 0, (0,0,0), 2)
    #cv2.imshow('Color Detection Result 4', detected4)
    print("Dark Gray :", area4)

    cv2.imshow('Original Image with Area at Centroid', originalImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

pathOfImage = './data/fig1.jpg'
calculateSegment(pathOfImage)
