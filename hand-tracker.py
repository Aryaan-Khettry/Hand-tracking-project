import mediapipe as mp #for actual hand detection (drawings)
import cv2 #importing opencv
import numpy as np #number operations
import uuid #to generate unique ID's
import os
mp_drawing.DrawingSpec()
os.mkdir('Images')
cap = cv2.VideoCapture(0) #capture webcam video(0 is webcam and 1+ denotes external cams
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened(): #test if camera is running in the users end
        ret, frame = cap.read() #get the next frame and the return value(True if frame is recieved)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #opencv reads in BGR color scheme        
        image = cv2.flip(image, 1) #flip the image to look more natural    
        image.flags.writeable = False     #to improve performance (pass by reference)   
        results = hands.process(image)    #create landmarks 
        image.flags.writeable = True               
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         ) #draw landmarks on screen
        cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid4())), image)
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

