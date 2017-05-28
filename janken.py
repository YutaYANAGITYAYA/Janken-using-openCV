# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

def show_result(x,y):
    print (u"チノちゃんの手: %s"%hand[x])
    print (u"　　あなたの手: %s"%hand[y])
    res = resMat[x][y]
    print (result[res])
    sumres[0 if res==1 else 1 if res==-1 else 2] += 1
    print (u"{0[0]}勝 {0[1]}敗\n".format(sumres))
    im = cv2.imread(os.path.join("data","%s.jpg"%res))
    cv2.imshow("janken", im)

def counter():
    ret, im = cam.read()
    im = cv2.bilateralFilter(im, 5, 16, 16)
    im = cv2.bilateralFilter(im, 5, 16, 16)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV)
    H = cv2.getTrackbarPos('Hue', 'CamImg')
    minH = max(0, H-5) if H<175 else 170
    maxH = min(180, H+5) if H>5 else 10
    mask = cv2.inRange(hsv, np.array([minH,31,31]), np.array([maxH,255,255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    count = 0
    if len(contours)>0:
        max_idx = np.argsort([len(cnt) for cnt in contours])[-1]
        cv2.drawContours(im, contours, max_idx, (255,0,0), 2)
        cnt = contours[max_idx]
        hull = cv2.convexHull(cnt, returnPoints = False)
        defects = cv2.convexityDefects(cnt, hull)
        for i in range(defects.shape[0]):
            s_idx,e_idx,f_idx,depth = defects[i,0]
            start = tuple(cnt[s_idx][0])
            end = tuple(cnt[e_idx][0])
            far = tuple(cnt[f_idx][0])
            dst = np.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)
            if 25<dst<175 and 20000<depth<50000:
                count += 1
                cv2.line(im,start,far,(0,255,0))
                cv2.line(im,far,end,(0,255,0))
                cv2.circle(im,far,3,(0,0,255),-1)
    
    cv2.imshow("CamImg", im)
    cv2.waitKey(1)
    return count

def nothing(x):
    pass

hand = {0:u"グー", 1:u"チョキ", 2:u"パー"}
result = {-1:u"チノちゃんの勝ち", 0:u"あいこ．．．", 1:u"あなたの勝ち"}
resMat = [[0,-1,1],[1,0,-1],[-1,1,0]]
sumres = [0,0,0]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
cam = cv2.VideoCapture(0)
mov = cv2.VideoCapture(os.path.join("data","chino.avi"))

cv2.namedWindow('CamImg')
cv2.createTrackbar('Hue', 'CamImg', 10, 180, nothing)

while cv2.waitKey(2000) != 27:
    mov.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while (1):
        ret, im = mov.read()
        if not ret:
            break
        im = cv2.resize(im, (854,480))
        cv2.imshow("janken", im)
        counter()
        cv2.waitKey(10)
    
    fing = sum([counter() for _ in range(10)])
    yy = 0 if fing<5 else 1 if fing<25 else 2
    cc = np.random.randint(3)
    im = cv2.imread(os.path.join("data","%s.png"%cc))
    im = cv2.resize(im, (854,480))
    cv2.imshow("janken", im)
    cv2.waitKey(1000)
    show_result(cc, yy)

cam.release()
cv2.destroyAllWindows()
