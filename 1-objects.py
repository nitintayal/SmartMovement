#
# url = "https://www.youtube.com/watch?v=WxgtahHmhiw"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")




# OpenCV Python program to detect cars in video frame
# import libraries of python OpenCV
import cv2
import glob
# for img in glob.glob("Images/*.jpg"):
#     cv_img = cv2.imread(img)

# capture frames from a video
cap = cv2.VideoCapture('vehicle.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 5)
pathIn = 'Images/'
platesPathIn = 'Plates/'
# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cars.xml')
plate_cascade = cv2.CascadeClassifier('plate/russian_plate_number.xml')
i=0
j=0
# loop runs if capturing has been initialized.
while (cap.isOpened()):
    # reads frames from a video
    ret, frames = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    x=1
    y=25
    h=1050
    w=2190
    roi = frames[y:y+h, x:x+w]
    cv2.rectangle(frames,(x,y),(x+w,y+h),(255,0,255),2)


    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(roi, 1.1, 2, minSize=(250,250))


    for (x,y,w,h) in cars:
        cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,255),2)
        crop_img = roi[y:y+h, x:x+w]
        cv2.imwrite(pathIn+str(i)+'.jpg',crop_img)
        i+=1

   # Display frames in a window
    if ret == True:
       # img = cv2.imread(frames)
       # y=0
       # x=0
       # h=100
       # w=200
       # crop_img = img[y:y+h, x:x+w]
       # cv2.vertices(frames,(0,600),(50,400),(100, 400),(2512,600),(100, 0, 0))
           #define the screen resulation
       #cv2.WINDOW_NORMAL makes the output window resizealbe
       cv2.namedWindow('CCTV CAMERA', cv2.WINDOW_NORMAL)

       #resize the window according to the screen resolution

       cv2.imshow('CCTV CAMERA', frames)
       cv2.resizeWindow('CCTV CAMERA', 1000, 700)
       if cv2.waitKey(25) & 0xFF == ord('q'):
           break

    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break
cv2.imwrite(pathIn+'1.png',frames)
# De-allocate any associated memory usage
capture.release()
cv2.destroyAllWindows()
