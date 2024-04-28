import cv2 
import numpy as np

cap = cv2.VideoCapture(0) 

while(cap.isOpened()): 
    # Capture frames from the video 
    ret, frame = cap.read() 

    if ret:
        # Display the resulting frame 
        cv2.imshow('frame', frame) 
        print(frame)

        # creating 'q' as the quit button for the video 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    else:
        print("Error: Failed to capture frame")

# release the cap object 
cap.release() 
# close all windows 
cv2.destroyAllWindows()
