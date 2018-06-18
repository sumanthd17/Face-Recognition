import cv2

vidcap = cv2.VideoCapture('justice_league.mp4');
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)
success,image = vidcap.read()
count = 0
success = True

counter = 0
while success:
    success,image = vidcap.read()
    print('read a new frame:',success)

    if count%240 == 0 :
         #cv2.imwrite('frame%d.jpg'%count,image)
         print('success')
         counter += 1
    count+=1
print(counter)