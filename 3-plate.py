import cv2
import imutils
import numpy as np
import pytesseract
import os

directory = 'Images/'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
plate_cascade = cv2.CascadeClassifier('plate/russian_plate_number.xml')
pathIn = 'Plates/'
f = open("Plates/plates.txt", "a")

i=0
img = cv2.imread(os.path.join(directory, "26.jpg"),cv2.IMREAD_COLOR)
cv2.imshow('Vehicle Image',img)
cv2.waitKey(0)
# cv2.imshow('car1',img)
# cv2.waitKey(0)
cars = plate_cascade.detectMultiScale(img, 1.1, 1, minSize=(50,49))
for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
    crop_img = img[y:y+h, x:x+w]


cv2.imshow('Detected ROI',img)
cv2.waitKey(0)



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 20, 20, 20)
cv2.imshow('gray bilateralFilter',gray)
cv2.waitKey(0)


edged = cv2.Canny(gray, 30, 200)
cv2.imshow('edged',edged)
cv2.waitKey(0)
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None


for c in contours:

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
     detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("programming_fever's License Plate Recognition\n")
print("Detected license plate Number is:",text)
img = cv2.resize(img,(500,300))
Cropped = cv2.resize(Cropped,(400,200))
cv2.imshow('car',img)
cv2.imshow('Cropped',Cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
