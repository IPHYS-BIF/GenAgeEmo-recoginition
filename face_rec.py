import cv2
from keras.models import model_from_json
from keras.utils import img_to_array
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = model_from_json(open("E:/Hadraba/face_rec/facial_expression_model_structure.json", "r").read())
model.load_weights('E:/Hadraba/face_rec/facial_expression_model_weights.h5')
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

 
while(True):
    ret, img = cap.read()    #apply same face detection procedures

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

        img_pixels = img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)

        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
        predictions = model.predict(img_pixels) #store probabilities of 7 expressions
        print(predictions)
		# find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])
		
        emotion = emotions[max_index]
		
		#write emotion text above rectangle
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow('faces', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break
 
cap.release()
cv2.destroyAllWindows()