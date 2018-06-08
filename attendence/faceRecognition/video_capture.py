import cv2

vidcap = cv2.VideoCapture('justice_league.mp4');
success,image = vidcap.read()
count = 0
success = True

while success:
    success,image = vidcap.read()
    print('read a new frame:',success)
    if count%10 == 0 :
         cv2.imwrite('frame%d.jpg'%count,image)
         print('success')
    count+=1