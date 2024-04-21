import cv2 
import numpy as np

cap = cv2.VideoCapture(0) 

while(True): 
    # Capture frames from the video 
    ret, frame = cap.read() 

    # describe the type of font to be used
    font = cv2.FONT_HERSHEY_SIMPLEX 

    # Calculate text size to determine its width and height
    text = 'TEXT ON VIDEO'
    text_size, _ = cv2.getTextSize(text, font, 1, 2)

    # Calculate position to center the text on the frame
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2

    # Draw text on the frame
    cv2.putText(frame, 
                text, 
                (text_x, text_y), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)

    # Display the resulting frame 
    cv2.imshow('frame', frame) 

    # creating 'q' as the quit button for the video 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# release the cap object 
cap.release() 
# close all windows 
cv2.destroyAllWindows()
