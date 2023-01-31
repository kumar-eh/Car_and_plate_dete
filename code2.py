
import cv2

# capture frames from a video

cap = cv2.VideoCapture("roi_video.mp4")
vid = cv2.VideoCapture("vidoes/full_video.mp4")
currentframe =0 

# Trained XML classifiers describes some features of some object we want to detect
number_cascade = cv2.CascadeClassifier('number_plate_cascade.xml')
car_cascade = cv2.CascadeClassifier('car_cascade.xml')

# loop runs if capturing has been initialized.
while True:
    # reads frames from a video
    ret, frames = cap.read()
    ret1 , frame1 = vid.read()
    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # Detects cars of different sizes in the input image
    plate = number_cascade.detectMultiScale( gray, 1.1, 7)
    car = car_cascade.detectMultiScale(gray1 , 1.1 ,7)
    # To draw a rectangle in each plate
    # for (x,y,w,h) in plate:
    #     cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
    #     # Display frames in a window
    #     cv2.imwrite('plates\plate' + str(currentframe) + '.jpg' ,frames )
    #     currentframe+=1
    #     cv2.imshow('Plate Detection', frames)

    for (x,y,w,h) in car:
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2)
        # Display frames in a window
        if(w > 214 and h>214 ):
            cv2.imwrite('cars\car' + str(currentframe) + '.jpg' ,frame1 )
            currentframe+=1
        cv2.imshow('Car Detection', frame1)
    # Wait for Enter key to stop
    if cv2.waitKey(33) == 13:
        break

cv2.destroyAllWindows()