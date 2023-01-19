import cv2

#create a cascade classifier object
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#load image with face which needs to be detected, and create a gray scale image
img=cv2.imread("test1.jpg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Detect Multi Scale method to get coordinates of the detected face/s.
faces=face_cascade.detectMultiScale(gray_img,
scaleFactor=1.05,
minNeighbors=23)

#draw colorful rectangle of the 4 points detected with Multi Scale method
for x,y,w,h in faces:
    img=cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 7)

#check if the detected face is a numpy array
print(type(faces))
print(faces)

#resize image in ratio before showing it
resized=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

#show faces detected for 5 seconds and close the window
cv2.imshow("Gray", resized)
cv2.waitKey(5000)
cv2.destroyAllWindows()
