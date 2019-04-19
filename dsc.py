import cv2
from darkflow.net.build import TFNet
import numpy as np

options={
	'model':'cfg/yolo.cfg',
	'load':'bin/yolo.weights',
    'threshold':0.2,  #if return_predict() is > than threshold then it'll return otherwise it wont #threshold and gpu should be an integer value
    'gpu':0.0, #this means we're not using d gpu i.e. we r using cpu only
}
tfnet=TFNet(options)  #tfnet is object of TFNet

colors=[tuple(255*np.random.rand(3)) for x in range(10)]
capture=cv2.VideoCapture("football.mp4")

while True:
    ret,frame=capture.read()
    if ret:  #if ret !=0 basically
        results=tfnet.return_predict(frame) #since results is from frame hence we'll put d rect and text on frame
        for color,result in zip(colors,results):  #zip() returns ith argument to ith variable
            tl=(result['topleft']['x'],result['topleft']['y'])
            br=(result['bottomright']['x'],result['bottomright']['y'])
            label=result['label']
            confidence=result['confidence']
            text='{}:{:.0f}%'.format(label,confidence*100)  # d colon after { and before .0f...it means its after d first {something}
            frame=cv2.rectangle(frame,tl,br,color,3)  #(img,pt1,pt2,colour,thickness)
            frame=cv2.putText(frame,text,tl,cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
            #type of case=camel case  #(img,text var,origin pt=tl,font face,font scale,font colour(r,g,b),thickness)
        cv2.imshow('OUTPUT',frame) #like image show
    if cv2.waitKey(1) & 0xFF==ord('q'):  #press a specific key to exit basically  #ord gives u the unicode # & is bitwise..its checking if q is pressed #1 represents that true so now exit..i.e. q has been pressed
        break 
capture.release()
cv2.destroyAllWindows