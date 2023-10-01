import cv2
import numpy as np


def calculateAreaPercentage(pathOfImage1, pathOfImage2):
    
    originalImage1 = cv2.imread(pathOfImage1)
    image1 = originalImage1.copy()
    image2 = originalImage1.copy()
    image3 = originalImage1.copy()
    image4 = originalImage1.copy()
    hsvImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    
    originalImage2 = cv2.imread(pathOfImage2)
    secondimage1 = originalImage2.copy()
    secondimage2 = originalImage2.copy()
    secondimage3 = originalImage2.copy()
    secondimage4 = originalImage2.copy()
    hsvImage2 = cv2.cvtColor(secondimage1, cv2.COLOR_BGR2HSV)


    #Yellow Portion
    yellowLower = np.array([20, 100, 100])
    yellowUpper = np.array([30, 255, 255])
    mask1 = cv2.inRange(hsvImage1, yellowLower, yellowUpper)
    detected1 = cv2.bitwise_and(image1, image1, mask=mask1)
    cnts1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area1=0
    for c in cnts1:
        area1 += cv2.contourArea(c)
    print("Yellow Area :",area1)

    yellowLower2 = np.array([20, 100, 100])
    yellowUpper2 = np.array([30, 255, 255])
    secmask1 = cv2.inRange(hsvImage2, yellowLower2, yellowUpper2)
    secdetected1 = cv2.bitwise_and(secondimage1, secondimage1, mask=secmask1)
    seccnts1, _ = cv2.findContours(secmask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    secArea1=0
    for c in seccnts1:
        secArea1 += cv2.contourArea(c)
    print("Yellow 2 Area :",secArea1)
    
    redAreaPercentage1=((area1-secArea1)/area1)*100
    print("Red Area Percentage 1 :",redAreaPercentage1)
    
    
    #Light Gray Portion
    lightLower = np.array([0, 0, 220])
    lightUpper = np.array([180, 30, 250])
    mask2 = cv2.inRange(hsvImage1, lightLower, lightUpper)
    detected2 = cv2.bitwise_and(image2, image2, mask=mask2)
    cnts2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area2=0
    for c in cnts2:
        area2 += cv2.contourArea(c)
    print("\nLight Gray Area :",area2)
    
    lightLower2 = np.array([0, 0, 220])
    lightUpper2 = np.array([180, 30, 250])
    secmask2 = cv2.inRange(hsvImage2, lightLower2, lightUpper2)
    secdetected2 = cv2.bitwise_and(secondimage2, secondimage2, mask=secmask2)
    secCnts2, _ = cv2.findContours(secmask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    secArea2=0
    for c in secCnts2:
        secArea2 += cv2.contourArea(c)
    print("Light Gray 2 Area :",secArea2)
    
    redAreaPercentage2=((area2-secArea2)/area2)*100
    print("Red Area Percentage 2 :",redAreaPercentage2)
    
    
    #Gray Portion
    grayLower = np.array([0, 0, 200])
    grayUpper = np.array([220, 0, 250])
    mask3 = cv2.inRange(hsvImage1, grayLower, grayUpper)
    detected3 = cv2.bitwise_and(image3, image3, mask=mask3)
    cnts3, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area3=0
    for c in cnts3:
        area3 += cv2.contourArea(c)
    print("\nGray Area :",area3)
    
    grayLower2 = np.array([0, 0, 200])
    grayUpper2 = np.array([220, 0, 250])
    secmask3 = cv2.inRange(hsvImage2, grayLower2, grayUpper2)
    secdetected3 = cv2.bitwise_and(secondimage3, secondimage3, mask=mask3)
    secCnts3, _ = cv2.findContours(secmask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    secArea3=0
    for c in secCnts3:
        secArea3 += cv2.contourArea(c)
    print("Gray Area :",secArea3)
    
    redAreaPercentage3=((area3-secArea3)/area3)*100
    print("Red Area Percentage 3 :",redAreaPercentage3)
    
    
    #Dark Gray Portion
    darkLower = np.array([0, 0, 100])
    darkUpper = np.array([180, 30, 200])
    mask4 = cv2.inRange(hsvImage1, darkLower, darkUpper)
    detected4 = cv2.bitwise_and(image4, image4, mask=mask4)
    cnts4, _ = cv2.findContours(mask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area4=0
    for c in cnts4:
        area4 += cv2.contourArea(c)
    print("\nDark Gray Area :", area4)

    darkLower2 = np.array([0, 0, 100])
    darkUpper2 = np.array([180, 30, 200])
    secmask4 = cv2.inRange(hsvImage2, darkLower2, darkUpper2)
    detected4 = cv2.bitwise_and(secondimage4, secondimage4, mask=secmask4)
    secCnts4, _ = cv2.findContours(secmask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    secArea4=0
    for c in secCnts4:
        secArea4 += cv2.contourArea(c)
    print("Dark Gray Area :", secArea4)

    redAreaPercentage4=((area4-secArea4)/area4)*100
    print("Red Area Percentage 4 :",redAreaPercentage4)


#Main Function
pathOfImage1 = './data/fig1.jpg'
pathOfImage2 = './data/fig2.jpg'
calculateAreaPercentage(pathOfImage1, pathOfImage2)
