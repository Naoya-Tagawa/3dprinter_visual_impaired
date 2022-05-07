import difflib
import time
import cv2
import numpy as np
import glob
from natsort import natsorted
import tkinter as tk
import tkinter.ttk as ttk
import threading
from PIL import Image
import pyttsx3 
from dictionary_word import speling
import difflib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
def cut_blue_img(img):
    c_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #ブルーに近いものを切り抜く
    average_color_per_row = np.average(c_img,axis=0)
    average_color = np.average(average_color_per_row,axis=0)
    average_color = np.uint8(average_color)
    #ブルーの最小値
    blue_min = np.array([100,130,180],np.uint8)
    #ブルーの最大値
    blue_max = np.array([120,255,255],np.uint8)
    threshold_blue_img = cv2.inRange(img_hsv,blue_min,blue_max)
    #threshold_blue_img = cv2.cvtColor(threshold_blue_img,cv2.COLOR_GRAY2RGB)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close_img = cv2.morphologyEx(threshold_blue_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    #文字の部分を塗りつぶす 
    cnts = cv2.findContours(close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(close_img, cnts, [255,255,255])
    plt.imshow(close_img)
    plt.show()
    return close_img

def points_extract(img):
    #コーナー検出
    dst = cv2.goodFeaturesToTrack(img,25,0.01,10)
    dst = np.int0(dst)
    #コーナーの中でx座標が最小、最大
    min_p= np.min(dst,axis=0)
    max_p = np.max(dst,axis=0)
    
    mi_x =[]
    ma_x = []

    #mi_x.append([min_p[0,0],min_p[0,1]])
    #ma_x.append([max_p[0,0],max_p[0,1]])
    for i in dst:
        x,y = i.ravel()
    
        if (x-min_p[0,0]) <=10:
                mi_x.append([x,y])
            
        if (max_p[0,0]-x) <=10:
                ma_x.append([x,y])
        #cv2.circle(c_img,(x,y),3,255,1)
    #cv2.circle(c_img,(591,139),50,255,-1)

    p1 =np.zeros(2)
    p2 =np.zeros(2)
    p3 =np.zeros(2)
    p4 =np.zeros(2)

    p =sorted(mi_x,key = lambda x:x[1])
    #左上
    p1 = p[0]
    #左下
    p2 = p[-1]
    p = sorted(ma_x,key = lambda x:x[1])
    #右上
    p3 = p[0]
    #右下
    p4 = p[-1]
    return p1,p2,p3,p4

def syaei(img1,p1,p2,p3,p4):
    #座標
    #p1 左上
    #p2 左下
    #p3 右上
    #p4 右下
    #幅
    w = np.linalg.norm(p3[0]-p1[0])
    w = math.floor(w)
    h = np.linalg.norm(p2[1]-p1[1])
    h = math.floor(h)
    #pts1はカードの4辺、pts2は変換後の座標
    pts1 = np.float32([p1,p3,p2,p4])
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    #射影変換を実施
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img1, M, (w, h))
    return dst
#縦方向のProjection profileを得る
def Projection_H(img, height, width):
    array_H = np.zeros(height)
    for i in range(height):
        total_count = 0
        for j in range(width):
            temp_pixVal = img[i, j]
            if (temp_pixVal == 0):
                total_count += 1
        array_H[i] = total_count
 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_axis = np.arange(height)
    ax.barh(x_axis, array_H)
    fig.savefig("hist_H.png")
 
    return array_H
 
 #横方向のProjection profileを得る
def Projection_V(img, height, width):
    array_V = np.zeros(width)
    for i in range(width):
        total_count = 0
        for j in range(height):
            temp_pixVal = img[j, i]
            if (temp_pixVal == 0):
                total_count += 1
        array_V[i] = total_count
 
 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_axis = np.arange(width)
    ax.bar(x_axis, array_V)
    fig.savefig("hist_V.png")
 
    return array_V
 
 #Projection profileから縦方向の座標を得る
def Detect_HeightPosition(H_THRESH, height, array_H):
    lower_posi = i
    upper_posi = i
    for i in range(height):
        val = array_H[i]
        if (val > H_THRESH):
            lower_posi = i
            break
 
    for i in reversed(range(height)):
        val = array_H[i]
        if (val > H_THRESH):
            upper_posi = i
            break
 
    return lower_posi, upper_posi
 
 
def Detect_WidthPosition(W_THRESH, width, array_V):
    char_List = np.array([])
 
    flg = False
    posi1 = 0
    posi2 = 0
    for i in range(width):
        val = array_V[i]
        if (flg==False and val < W_THRESH):
            flg = True
            posi1 = i
 
        if (flg == True and val >= W_THRESH):
            flg = False
            posi2 = i
            char_List = np.append(char_List, posi1)
            char_List = np.append(char_List, posi2)
 
    return char_List
 
 
if __name__ == "__main__":
    # input image
    img = cv2.imread("./camera1/camera12.jpg")
    #対象画像をロード
    #青い部分のみを二値化
    close_img = cut_blue_img(img)
    #コーナー検出
    p1,p2,p3,p4 = points_extract(close_img)
    #コーナーに従って画像の切り取り
    #img_k = img[p1[1]:p2[1],p2[0]:p3[0]]
    #射影変換
    syaei_img = syaei(img,p1,p2,p3,p4)

    img_k = syaei_img[13:54,:]
    #cv2.imwrite(r'C:\Users\Naoya Tagawa\OneDrive\s.jpg',syaei_img)
    s_img = cv2.cvtColor(syaei_img,cv2.COLOR_BGR2RGB)
    plt.imshow(img_k)
    plt.show()

    # convert gray scale image
    gray_img = cv2.cvtColor(img_k, cv2.COLOR_RGB2GRAY)
 
    # black white
    ret, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    height, width = bw_img.shape
 
    # create projection distribution
    array_H = Projection_H(bw_img, height, width)
    array_V = Projection_V(bw_img, height, width)
 
    # detect character height position
    H_THRESH = 5
    lower_posi, upper_posi = Detect_HeightPosition(H_THRESH, height, array_H)
 
    # detect character width position
    W_THRESH = max(array_V)
    char_List = Detect_WidthPosition(W_THRESH, width, array_V)
    print(array_V)
    print(array_H)
    print(char_List)
    print(upper_posi)
    print(lower_posi)
    # draw image
    if (len(char_List) % 2) == 0:
        print("Succeeded in character detection")
        for i in range(0, (len(char_List)-1), 2):
            img_k = cv2.rectangle(img_k, (int(char_List[i]), int(upper_posi)), (int(char_List[i+1]), int(lower_posi)), (0,0,255), 2)
        cv2.imwrite("result.jpg", img_k)
        
    else:
        print("Failed to detect characters")