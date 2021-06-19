import cv2
import numpy as np

cap =cv2.VideoCapture(0) 

face_cascade=cv2.CascadeClassifier("C://Users//gokul//OneDrive//Desktop//python//eyes detection//haarcascade_frontalface_default.xml")

skip=0
face_data=[]
dataset_path="C://Users//gokul//OneDrive//Desktop//project//data//"

file_name=input("enter the name: ")
while True:

    ret,frame=cap.read()

    if ret==False:
        continue
    #cv2.imshow("frame",frame)
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])

    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        offset=10
        facesec=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        facesec=cv2.resize(facesec,(100,100))
        
        skip+=1
        
        if skip%10==0:
            face_data.append(facesec)
            print(len(face_data))


    cv2.imshow("frame",frame)
    cv2.imshow("facesection",facesec)
    #if (skip%10==0):
     #   pass











    key_pressed= cv2.waitKey(1) &0xFF
    if key_pressed ==ord('q'):
        break

face_data=np.asarray(face_data)
face_data=face_data.reshape(face_data.shape[0],-1)
print(face_data.shape)

np.save(dataset_path+file_name+'.npy',face_data)
print("data success")

cap.release()
cv2.destroyAllWindows()    
