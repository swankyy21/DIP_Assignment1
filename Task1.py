import cv2

image = cv2.imread('./data/rect1.jpg')
cv2.imshow('Original Image', image)
resultImage = image.copy()
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in  contours:
     x, y, w, h = cv2.boundingRect(c)
     length = max(w, h)
     width = min(w, h)

     topLeft = (x, y)
     topRight = (x + w, y)
     bottomLeft = (x, y + h)
     bottomRight = (x + w, y + h)

     cx = (topLeft[0]+topRight[0]+bottomLeft[0]+bottomRight[0]) / 4
     cy = (topLeft[1]+topRight[1]+bottomLeft[1]+bottomRight[1]) / 4

if length!=width:
     print('Given figure is a Rectangle')
else:
     print('Given figure is a Square')

print(f'Perimeter of the shape is {(2*length) + (2*width)}')
print(f'Centroid : {(cx, cy)}')


cv2.circle(resultImage, (int(cx), int(cy)), 2, (0, 0, 255), -1) 
cv2.drawContours(resultImage, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contoured Image', resultImage)
cv2.waitKey(0)
cv2.destroyAllWindows()