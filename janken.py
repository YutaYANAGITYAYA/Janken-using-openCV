# -*- coding: utf-8 -*-
import cv2
import numpy as np

def show_result(x,y):
    print u"チノちゃんの手: %s"%hand[x]
    print u"　　あなたの手: %s"%hand[y]
    res = resMat[x][y]
    print result[res]
    img = cv2.imread("data\%s.jpg"%res)
    cv2.imshow("janken",img)

def counter():
    h = cv2.getTrackbarPos('Hue', 'CamImg')
    lower = h-5 if h-5>0 else 0
    upper = h+5 if h+5<180 else 180

    # 取得した画像から肌色抽出
    ret, im = cam.read()
    im = cv2.bilateralFilter(im, 7, 21, 21)
    im = cv2.bilateralFilter(im, 7, 21, 21)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV) 
    mask = cv2.inRange(hsv, np.array([lower,31,31]), np.array([upper,255,255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 最大輪郭のものに対して指の数をカウント
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    count = 0
    if len(contours)>0:
        max_idx = np.argsort([len(cnt) for cnt in contours])[-1]
        cv2.drawContours(im, contours, max_idx, (255,0,0), 2)
        cnt = contours[max_idx]
        hull = cv2.convexHull(cnt, returnPoints = False)
        defects = cv2.convexityDefects(cnt, hull)
        for i in xrange(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            dst = np.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)
            if 25<dst<150 and 20000<d<50000:
                count += 1
                cv2.line(im,start,far,(0,255,0))
                cv2.line(im,far,end,(0,255,0))
                cv2.circle(im,far,2,(0,0,255),-1)
    
    cv2.imshow("CamImg",im)
    cv2.waitKey(1)
    return count

def nothing(x):
    pass


hand = {0:u"グー", 1:u"チョキ", 2:u"パー"}
result = {-1:u"チノちゃんの勝ち\n", 0:u"あいこ．．．\n", 1:u"あなたの勝ち\n"}
resMat=[[0,-1,1],[1,0,-1],[-1,1,0]]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

cam = cv2.VideoCapture(0)
mov = cv2.VideoCapture("data\chi.avi")

cv2.namedWindow('CamImg')
cv2.createTrackbar('Hue', 'CamImg', 10, 180, nothing)
cv2.waitKey(1)

while (cv2.waitKey(2000)!=27):
    mov.set(cv2.CAP_PROP_POS_FRAMES,0)
    while (1):
        ret, img = mov.read()
        if not ret:
            break
        cv2.imshow("janken",img)
        counter()
        cv2.waitKey(33)
    
    fing = 0
    for i in xrange(10):
        fing += counter()
    yy = 0 if fing<5 else 1 if fing<25 else 2
    cc = np.random.randint(3)
    img = cv2.imread("data\%s.png"%cc)
    cv2.imshow("janken",img)
    cv2.waitKey(1000)
    show_result(cc,yy)


cam.release()
cv2.destroyAllWindows()