import cv2

vc = cv2.VideoCapture('/data/zhongchongyang/motiondiffuse/clockwise.mp4')
n=1

if vc.isOpened():
    rval,frame=vc.read()
else:
    rval=False

timeF=1
i=0
while rval:
    rval,frame=vc.read()
    if (n % timeF == 0):
        i +=1
        cv2.imwrite('/data/zhongchongyang/motiondiffuse/wave/{}.jpg'.format(i),frame)
    n = n+1
    cv2.waitKey(1)
vc.release