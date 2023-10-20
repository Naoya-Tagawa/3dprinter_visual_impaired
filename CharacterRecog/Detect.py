from hashlib import new
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import time
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from natsort import natsorted
import threading
from PIL import Image
import difflib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylsd.lsd import lsd
import os
import pyocr
import math
def mask_make(blue_threshold_present_img):
    kernel = np.ones((3,3),np.uint8)
    #hsvLower = np.array([100,130,180])
    hsvLower = np.array([70, 25, 25])    #s 抽出する色の下限(HSV)
    hsvUpper = np.array([255, 222, 255])    # 抽出する色の上限(HSV)
    hsv = cv2.cvtColor(blue_threshold_present_img, cv2.COLOR_BGR2HSV) # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    #result = cv2.bitwise_and(blue_threshold_present_img, blue_threshold_present_img, mask=hsv_mask) # 元画像とマスクを合成
    hsv_mask = cv2.medianBlur(hsv_mask,3)
    cv2.imwrite("m.png",hsv_mask)
    #mask_present_img2 = cv2.dilate(mask_present_img2,kernel)
    #ret, mask_present_img2 = cv2.threshold(hsv_mask,0,255,cv2.THRESH_OTSU)
    #mask_present_img2 = cv2.dilate(hsv_mask,kernel,iterations=1)
    mask_present_img2 = cv2.dilate(hsv_mask,kernel)
    #mask_present_img2 = hsv_mask
    #plt.imshow(mask_present_img2)
    #plt.show()
    cv2.imwrite("mask.png",mask_present_img2)

    height_present , width_present = mask_present_img2.shape
    
    array_present_H = Projection_H(mask_present_img2,height_present,width_present)
    presentH_THRESH = max(array_present_H)
    present_char_List = Detect_HeightPosition(presentH_THRESH,height_present,array_present_H)
    #print(present_char_List)
    present_char_List = np.reshape(present_char_List,[int(len(present_char_List)/2),2])
    #present_char_List = image_processing.convert_1d_to_2d(present_char_List,2)
    return present_char_List , mask_present_img2

def make_char_list(mask_present_img2):
        height_present , width_present = mask_present_img2.shape
    
        array_present_H = Projection_H(mask_present_img2,height_present,width_present)
        presentH_THRESH = max(array_present_H)
        present_char_List = Detect_HeightPosition(presentH_THRESH,height_present,array_present_H)
        #print(present_char_List)
        present_char_List = np.reshape(present_char_List,[int(len(present_char_List)/2),2])
        #present_char_List = image_processing.convert_1d_to_2d(present_char_List,2)
        return present_char_List


def mask_make1(blue_threshold_present_img):
    kernel = np.ones((3,3),np.uint8)
    #hsvLower = np.array([100,130,180])
    hsvLower = np.array([70, 25, 25])    #s 抽出する色の下限(HSV)
    hsvUpper = np.array([255, 222, 255])    # 抽出する色の上限(HSV)
    hsv = cv2.cvtColor(blue_threshold_present_img, cv2.COLOR_BGR2HSV) # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    #result = cv2.bitwise_and(blue_threshold_present_img, blue_threshold_present_img, mask=hsv_mask) # 元画像とマスクを合成
    hsv_mask = cv2.medianBlur(hsv_mask,3)
    #cv2.imwrite("m.png",hsv_mask)
    #mask_present_img2 = cv2.dilate(mask_present_img2,kernel)
    #ret, mask_present_img2 = cv2.threshold(hsv_mask,0,255,cv2.THRESH_OTSU)
    #mask_present_img2 = cv2.dilate(hsv_mask,kernel,iterations=1)
    mask_present_img2 = cv2.dilate(hsv_mask,kernel)
    #mask_present_img2 = hsv_mask
    #plt.imshow(mask_present_img2)
    #plt.show()
    cv2.imwrite("mask.png",mask_present_img2)

    #height_present , width_present = mask_present_img2.shape
    return  mask_present_img2


#アオイ部分を切り抜く
def cut_blue_img1(img):
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
    dst = cv2.bitwise_and(img,img,mask = close_img)
    return dst
def cut_blue_img2(img):
    #γ変換の値
    gamma=0.24
    #γ変換の対応表を作る
    LUT_Table=np.zeros((256,1),dtype='uint8')
    for i in range(len(LUT_Table)):
        LUT_Table[i][0]=255*(float(i)/255)**(1.0/gamma)
    #γ変換をする
    img=cv2.LUT(img,LUT_Table)
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
    dst = cv2.bitwise_and(img,img,mask = close_img)
    return dst

def cut_blue_trans(img):
    #γ変換の値
    gamma=0.2
    #γ変換の対応表を作る
    LUT_Table=np.zeros((256,1),dtype='uint8')
    for i in range(len(LUT_Table)):
        LUT_Table[i][0]=255*(float(i)/255)**(1.0/gamma)
    #γ変換をする
    img=cv2.LUT(img,LUT_Table)
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
    
    return close_img

def cut_blue_trans2(img):
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
    #dst = cv2.bitwise_and(img,img,mask = close_img)
    return close_img

#コーナー検出
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
        cv2.circle(img,(x,y),3,255,1)
    #cv2.circle(c_img,(591,139),50,255,-1)
    #plt.imshow(img)
    #plt.show()
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


#コーナー検出

# 点p0に一番近い点を取得
def func_search_neighbourhood(p0, ps):
    L = np.array([])
    s = {}
    for i in range(ps.shape[0]):
        norm = np.sqrt( (ps[i][0] - p0[0])*(ps[i][0] - p0[0]) +
                        (ps[i][1] - p0[1])*(ps[i][1] - p0[1]) )
        #print(norm)
        if norm <= 10:
            s.setdefault(norm,i)
    new_d = sorted(s.items())
    if len(s) == 0:
        
        return p0
    else:
        return ps[new_d[0][1]]

def points_extract2(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("00",img)
    #cv2.waitKey(0)
    #コーナー検出
    #img_1 = cv2.Canny(img,50,150)
    #linesH = cv2.HoughLinesP(img,rho=1,theta = np.pi/360,threshold=50,minLineLength=50,maxLineGap=10)
    
    #コーナーの中でx座標が最小、最大

    #Smin_p= np.min(dst,axis=0)
    #max_p = np.max(dst,axis=0)
    linsl = lsd(img)
    #x1,y1の列のみ抽出
    xy1_line = linsl[:,0:2]
    #x2,y2の列のみ抽出
    xy2_line = linsl[:,2:4]

    la = np.vstack((xy1_line,xy2_line))
    
    la = np.asarray(la,dtype=int)
    mi_x =[]
    ma_x = []
    #print(linsl)
    #print(ly)
    #data = BytesIO(ly)
    x_point = la[:,0]
    la = la[np.argsort(x_point)]
    l_distance = [ math.sqrt(x*x + y*y)  for x,y in la]
    min_index = l_distance.index(min(l_distance))
    reverse_l_distance = l_distance[::-1]
    max_index = l_distance.index(max(reverse_l_distance))
    #print(ly)
    min_x = int(la[min_index][0])
    max_x = int(la[max_index][0])
    #print(min_x)
    #print(max_x)
    #print(int(la[min_index][1]))
    #print(int(la[max_index][1]))
    mi_x = [ [x,y] for x,y in la if x-min_x <=30]
    minmax_distance = [ math.sqrt(x*x + y*y)  for x,y in mi_x]
    minmax_index = minmax_distance.index(max(minmax_distance))
    #左下
    #print("左hした")
    #print(mi_x[minmax_index][0])
    #print(mi_x[minmax_index][1])
    mi_x = np.array(mi_x)
    mi_x = mi_x.ravel()
    ma_x = [ [x,y] for x,y in la if max_x-x <=30]
    maxmin_distance = [ math.sqrt(x*x + y*y)  for x,y in ma_x]
    maxmin_index = maxmin_distance.index(min(maxmin_distance))
    #print("右上")
    #print(ma_x[maxmin_index][0],ma_x[maxmin_index][1])
    ma_x = np.array(ma_x)
    ma_x = ma_x.ravel()
    mi_x = mi_x.reshape(int(len(mi_x)/2),2)
    ma_x = ma_x.reshape(int(len(ma_x)/2),2)
    
    ma_x = ma_x[np.argsort(ma_x[:,1])]
    mi_x = mi_x[np.argsort(mi_x[:,1])] 
    
    #左うえ
    #min_1 = [int(mi_x[0][0]),int(mi_x[0][1])]
    min_1 = [int(la[min_index][0]),int(la[min_index][1])]
    #print(int(la[max_index][1]))
    near_min_1 = func_search_neighbourhood(min_1,mi_x[1:])
    #右上
    #max_1 = [int(ma_x[0][0]),int(ma_x[0][1])]
    max_1 = [int(ma_x[maxmin_index][0]),int(ma_x[maxmin_index][1])]
    
    near_max_1 = func_search_neighbourhood(max_1,ma_x[1:])
    #左下
    #min_2 = [int(mi_x[-1][0]),int(mi_x[-1][1])]
    min_2 = [int(mi_x[minmax_index][0]),int(mi_x[minmax_index][1])]
    near_min_2 = func_search_neighbourhood(min_2,mi_x[:-1])
    #右下
    #max_2 = [int(ma_x[-1][0]),int(ma_x[-1][1])]
    max_2 = [int(la[max_index][0]),int(la[max_index][1])]
    near_max_2 = func_search_neighbourhood(max_2,ma_x[:-1])
    #print(near_max_2)
    #print(max_2)
    #ひだりうえ
    p1 = [int((min_1[0]+near_min_1[0])/2),int((min_1[1]+near_min_1[1])/2)]
    #左下
    p2 = [int((min_2[0]+near_min_2[0])/2),int((min_2[1]+near_min_2[1])/2)]
    #右上
    p3 = [int((max_1[0]+near_max_1[0])/2),int((max_1[1]+near_max_1[1])/2)]
    #右下
    p4 = [int((max_2[0]+near_max_2[0])/2),int((max_2[1]+near_max_2[1])/2)]
    if img[p1[1]][p1[0]] == 0:
        p1 = [int(min_1[0]),int(min_1[1])]
    if img[p2[1]][p2[0]] == 0:
        p2 = [int(min_2[0]),int(min_2[1])]
    if img[p3[1]][p3[0]] == 0:
        p3 = [int(max_1[0]),int(max_1[1])]
    if img[p4[1]][p4[0]] == 0:
        p4 = [int(max_2[0]),int(max_2[1])]
    #print(p1)
    #print(p2)
    #print(p3)
    #print(p4)
    #de_position = pd.read_csv(io.BytesIO(ly))
    #print(de_position)
# 小さい輪郭は誤検出として削除する
    #contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))

    # 輪郭を描画する。
    #cv2.drawContours(img2, contours, -1, color=(0, 0, 255), thickness=3)

    #plt.imshow(img2)
    #plt.show()
    return p1,p2,p3,p4
def points_extract1(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("00",img)
    #cv2.waitKey(0)
    #コーナー検出
    #img_1 = cv2.Canny(img,50,150)
    #linesH = cv2.HoughLinesP(img,rho=1,theta = np.pi/360,threshold=50,minLineLength=50,maxLineGap=10)
    
    #コーナーの中でx座標が最小、最大

    #Smin_p= np.min(dst,axis=0)
    #max_p = np.max(dst,axis=0)
    linsl = lsd(img)
    #x1,y1の列のみ抽出
    xy1_line = linsl[:,0:2]
    #x2,y2の列のみ抽出
    xy2_line = linsl[:,2:4]

    la = np.vstack((xy1_line,xy2_line))
    
    la = np.asarray(la,dtype=int)
    mi_x =[]
    ma_x = []
    #print(linsl)
    #print(ly)
    #data = BytesIO(ly)
    x_point = la[:,0]
    la = la[np.argsort(x_point)]
    
    #print(ly)
    min_x = int(la[0][0])
    max_x = int(la[-1][0])
    mi_x = [ [x,y] for x,y in la if x-min_x <=10]
    mi_x = np.array(mi_x)
    mi_x = mi_x.ravel()
    ma_x = [ [x,y] for x,y in la if max_x-x <=30]
    ma_x = np.array(ma_x)
    ma_x = ma_x.ravel()
    mi_x = mi_x.reshape(int(len(mi_x)/2),2)
    ma_x = ma_x.reshape(int(len(ma_x)/2),2)
    #print(ma_x)
    #print(mi_x)
    ma_x = ma_x[np.argsort(ma_x[:,1])]
    mi_x = mi_x[np.argsort(mi_x[:,1])] 
    #print(ma_x)
    #print(mi_x)
    #左うえ
    min_1 = [int(mi_x[0][0]),int(mi_x[0][1])]
    near_min_1 = func_search_neighbourhood(min_1,mi_x[1:])
    #右上
    max_1 = [int(ma_x[0][0]),int(ma_x[0][1])]
    near_max_1 = func_search_neighbourhood(max_1,ma_x[1:])
    #左下
    min_2 = [int(mi_x[-1][0]),int(mi_x[-1][1])]
    near_min_2 = func_search_neighbourhood(min_2,mi_x[:-1])
    #右下
    max_2 = [int(ma_x[-1][0]),int(ma_x[-1][1])]
    
    near_max_2 = func_search_neighbourhood(max_2,ma_x[:-1])
    #print(near_max_2)
    #print(max_2)
    #ひだりうえ
    p1 = [int((min_1[0]+near_min_1[0])/2),int((min_1[1]+near_min_1[1])/2)]
    #左下
    p2 = [int((min_2[0]+near_min_2[0])/2),int((min_2[1]+near_min_2[1])/2)]
    #右上
    p3 = [int((max_1[0]+near_max_1[0])/2),int((max_1[1]+near_max_1[1])/2)]
    #右下
    p4 = [int((max_2[0]+near_max_2[0])/2),int((max_2[1]+near_max_2[1])/2)]
    if img[p1[1]][p1[0]] == 0:
        p1 = [int(min_1[0]),int(min_1[1])]
    if img[p2[1]][p2[0]] == 0:
        p2 = [int(min_2[0]),int(min_2[1])]
    if img[p3[1]][p3[0]] == 0:
        p3 = [int(max_1[0]),int(max_1[1])]
    if img[p4[1]][p4[0]] == 0:
        p4 = [int(max_2[0]),int(max_2[1])]
    #print(p1)
    #print(p2)
    #print(p3)
    #print(p4)
    #de_position = pd.read_csv(io.BytesIO(ly))
    #print(de_position)
# 小さい輪郭は誤検出として削除する
    #contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))

    # 輪郭を描画する。
    #cv2.drawContours(img2, contours, -1, color=(0, 0, 255), thickness=3)

    #plt.imshow(img2)
    #plt.show()
    return p1,p2,p3,p4
    

def projective_transformation(img1,p1,p2,p3,p4):
    #座標
    #p1 左上
    #p2 左下
    #p3 右上
    #p4 右下
    #幅
    #print(p3)
    #print(p1)
    w = np.linalg.norm(p3[0]-p1[0])
    w = math.floor(w)
    h = np.linalg.norm(p2[1]-p1[1])
    h = math.floor(h)
    #pts1はカードの4辺、pts2は変換後の座標
    pts1 = np.float32([p1,p3,p2,p4])
    #print(pts1)
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    #射影変換を実施
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img1, M, (w, h))
    #print(M)
    return dst

def projective_transformation2(img1,p1,p2,p3,p4):
    #座標
    #p1 左上
    #p2 左下
    #p3 右上
    #p4 右下
    #幅
    #print(p3)
    #print(p1)
    w = np.linalg.norm(p3[0]-p1[0])
    w = math.floor(w)
    h = np.linalg.norm(p2[1]-p1[1])
    h = math.floor(h)
    #pts1はカードの4辺、pts2は変換後の座標
    pts1 = np.float32([p1,p3,p2,p4])
    #print(pts1)
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    #射影変換を実施
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img1, M, (w, h))
    #print(M)
    return dst,M
#縦方向のProjection profileを得る
def Projection_H(img,h,w):
    array_H = np.full(h,w)
    count = [np.count_nonzero(img[i:i+1,]) for i in range(h)]
    count = np.asarray(count,dtype=int)
    array_H = array_H - count
    #array_H = np.zeros(h)
    #for i in range(h):
    #    at = img[i:i+1,]
    #    total_count = w - np.count_nonzero(at)
    #    array_H[i] = total_count
    
    return array_H
#横方向のProjection profileを得る
def Projection_V(img,h,w):
    array_V = np.full(w,h)
    count = [np.count_nonzero(img[:,i:i+1]) for i in range(w)]
    count = np.asarray(count,dtype=int)
    array_V = array_V - count
    return array_V
 
#Projection profileから縦方向の座標を得る
def Detect_HeightPosition(H_THRESH, height, array_H):
    char_List = np.array([])
 
    flg = False
    posi1 = 0
    posi2 = 0
    for i in range(height):
        val = array_H[i]
        if (flg==False and val < H_THRESH):
            flg = True
            posi1 = i
 
        if (flg == True and val >= H_THRESH):
            flg = False
            posi2 = i
            char_List = np.append(char_List, posi1)
            char_List = np.append(char_List, posi2)
 
    return char_List

#Projection profileから横方向の座標を得る
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
def match(img_temp,label_temp,frame):
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    window_img = frame
    #sフレームの青い部分を二値化
    blue_threshold_img = cut_blue_img1(window_img)
    plt.imshow(blue_threshold_img)
    plt.show()
    #高さ、幅を保持
    #コーナー検出
    try:
        p1,p2,p3,p4 = points_extract1(blue_threshold_img)
    except TypeError:
        print("point")
        print("Screen cannot be detected")
        return [],[]

    #コーナーに従って画像の切り取り
    cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]]
    #射影変換
    syaei_img = projective_transformation(window_img,p1,p2,p3,p4)
    #対象画像をリサイズ
    syaei_resize_img = cv2.resize(syaei_img,dsize=(610,211))
    #対象画像をグレイスケール化
    gray_img = cv2.cvtColor(syaei_resize_img,cv2.COLOR_BGR2GRAY)
    #二値画像へ
    ret, img_mask = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)
    #img_mask = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,-3)
    #ノイズ除去
    img_mask = cv2.medianBlur(img_mask,3)
    #膨張化
    #img_mask = cv2.dilate(img_mask,kernel)
    #plt.imshow(img_mask)
    #plt.show()
    #高さ、幅を保持
    height,width = img_mask.shape
    #縦方向のProjection Profileを保持
    array_H = Projection_H(img_mask,height,width)
    #縦方向の最大値を保持
    H_THRESH = max(array_H)
    char_List1 = Detect_HeightPosition(H_THRESH,height,array_H)

    #if (len(char_List1) % 2) == 0:
        #print("Screen cannot be detected")
        #return [], []
        
    out_modify = "" #修正したテキスト
    output_text = [] #読み取ったテキスト
    s = {}
    new_d = {}
    out = "" #読み取ったテキスト
    #横方向にきって列ごとに保存
    img_h = [img_mask[int(char_List1[i]):int(char_List1[i+1]),:] for i in range(0,len(char_List1)-1,2)]
    height_h  = [img_h[i].shape[0] for i in range(len(img_h))]
    width_h  = [img_h[i].shape[1] for i in range(len(img_h))]
    array_V = [Projection_V(img_h[i],height_h[i],width_h[i]) for i in range(len(img_h))]
    W_THRESH = [max(array_V[i]) for i in range(len(array_V))]
    char_List2 = [Detect_WidthPosition(W_THRESH[i],width_h[i],array_V[i]) for i in range(len(W_THRESH))]
    i = 0
    temp_th = [ cv2.resize(img_temp[i],dsize=(26,36)) for i in range(len(img_temp))]
    #print(len(temp_th))

    for j in range(0,len(char_List2[0])-1,2):
        #end_time = time.perf_counter()
        #print(end_time-start_time)
        #一文字ずつ切り取る
        img_h1 = img_h[0]
        match_img = img_h1[:,int(char_List2[0][j])-1:int(char_List2[0][j+1])+1]
        #cv2.imwrite("match.jpg",match_img)
        try:
            match_img = cv2.resize(match_img,dsize=(26,36))
        except cv2.error:
            return [],[]
        height_m,width_m = match_img.shape
        match = [cv2.matchTemplate(match_img,temp_th[i],cv2.TM_CCORR_NORMED) for i in range(len(label_temp))]
        max_value = [cv2.minMaxLoc(match[i])[1] for i in range(len(label_temp))]
        max_index = np.argmax(max_value)
        max_v = max_value[max_index]
        #空白があるとき
        if max_v < 0.7:
            i += 1
            print("out!!")
            continue
        if (j != 0) & (char_List2[0][j] > (width_m + char_List2[0][j-1])):

            if (j+1) == len(char_List2[0])-1:
                out_modify = out_modify+ ' ' + label_temp[max_index]
                out = out + out_modify + ' '
                out_modify = ""
                i += 1
                continue
                #out_modify = speling.correct(out_modify)
                #out_modify += label_temp[new_d[0][1]]
            out_modify += ' '
                #out = out + out_modify
                #output_text.append(' ')
                #output_text.append(out_modify)
                #print(out_modify)
                #out_modify = ""
                

            #行の最後の時
        if (j+1) == len(char_List2[0])-1:
            out_modify = out_modify + label_temp[max_index]
            #out_modify = speling.correct(out_modify)
            out = out + out_modify + " "
            out_modify = ""
            i += 1
            continue
            #print(label_temp[new_d[0][1]])
        out_modify = out_modify + label_temp[max_index]
           # print(out_modify)
        i += 1
        continue
    
    for j in range(0,len(char_List2[1])-1,2):
        #end_time = time.perf_counter()
        #print(end_time-start_ti
        #一文字ずつ切り取る
        img_h1 = img_h[1]
        match_img = img_h1[:,int(char_List2[1][j])-1:int(char_List2[1][j+1])+1]
        #cv2.imwrite("match.jpg",match_img)
        try:
            match_img = cv2.resize(match_img,dsize=(26,36))
        except cv2.error:
            return [],[]
        height_m,width_m = match_img.shape
        match = [cv2.matchTemplate(match_img,temp_th[i],cv2.TM_CCORR_NORMED) for i in range(len(label_temp))]
        max_value = [cv2.minMaxLoc(match[i])[1] for i in range(len(label_temp))]
        max_index = np.argmax(max_value)
        max_v = max_value[max_index]
        #空白があるとき
        if max_v < 0.7:
            i += 1
            print("out!!")
            continue
        if (j != 0) & (char_List2[1][j] > (width_m + char_List2[1][j-1])):

            if (j+1) == len(char_List2[1])-1:
                out_modify = out_modify+ ' ' + label_temp[max_index]
                out = out + out_modify + ' '
                out_modify = ""
                i += 1
                continue
                #out_modify = speling.correct(out_modify)
                #out_modify += label_temp[new_d[0][1]]
            out_modify += ' '
                #out = out + out_modify
                #output_text.append(' ')
                #output_text.append(out_modify)
                #print(out_modify)
                #out_modify = ""

            #行の最後の時
        if (j+1) == len(char_List2[1])-1:
            out_modify = out_modify + label_temp[max_index]
            #out_modify = speling.correct(out_modify)
            out = out + out_modify + " "
            out_modify = ""
            i += 1
            continue
            #print(label_temp[new_d[0][1]])
        out_modify = out_modify + label_temp[max_index]
           # print(out_modify)
        i += 1
        continue
    
    for j in range(0,len(char_List2[2])-1,2):
        #end_time = time.perf_counter()
        #print(end_time-start_time)
        #一文字ずつ切り取る
        img_h1 = img_h[2]
        match_img = img_h1[:,int(char_List2[2][j])-1:int(char_List2[2][j+1])+1]
        #cv2.imwrite("match.jpg",match_img)
        try:
            match_img = cv2.resize(match_img,dsize=(26,36))
        except cv2.error:
            return [],[]
        height_m,width_m = match_img.shape
        match = [cv2.matchTemplate(match_img,temp_th[i],cv2.TM_CCORR_NORMED) for i in range(len(label_temp))]
        max_value = [cv2.minMaxLoc(match[i])[1] for i in range(len(label_temp))]
        max_index = np.argmax(max_value)
        max_v = max_value[max_index]
        #空白があるとき
        if max_v < 0.7:
            i += 1
            print("out!!")
            continue
        if (j != 0) & (char_List2[2][j] > (width_m + char_List2[2][j-1])):

            if (j+1) == len(char_List2[2])-1:
                out_modify = out_modify+ ' ' + label_temp[max_index]
                out = out + out_modify + ' '
                out_modify = ""
                i += 1
                continue
                #out_modify = speling.correct(out_modify)
                #out_modify += label_temp[new_d[0][1]]
            out_modify += ' '
                #out = out + out_modify
                #output_text.append(' ')
                #output_text.append(out_modify)
                #print(out_modify)
                #out_modify = ""
                

            #行の最後の時
        if (j+1) == len(char_List2[2])-1:
            out_modify = out_modify + label_temp[max_index]
            #out_modify = speling.correct(out_modify)
            out = out + out_modify + " "
            out_modify = ""
            i += 1
            continue
            #print(label_temp[new_d[0][1]])
        out_modify = out_modify + label_temp[max_index]
           # print(out_modify)
        i += 1
        continue

    for j in range(0,len(char_List2[3])-1,2):
        #end_time = time.perf_counter()
        #print(end_time-start_ti
        #一文字ずつ切り取る
        img_h1 = img_h[3]
        match_img = img_h1[:,int(char_List2[3][j])-1:int(char_List2[3][j+1])+1]
        #cv2.imwrite("match.jpg",match_img)
        try:
            match_img = cv2.resize(match_img,dsize=(26,36))
        except cv2.error:
            return [],[]
        height_m,width_m = match_img.shape
        match = [cv2.matchTemplate(match_img,temp_th[i],cv2.TM_CCORR_NORMED) for i in range(len(label_temp))]
        max_value = [cv2.minMaxLoc(match[i])[1] for i in range(len(label_temp))]
        max_index = np.argmax(max_value)
        max_v = max_value[max_index]
        #空白があるとき
        if max_v < 0.7:
            i += 1
            print("out!!")
            continue
        if (j != 0) & (char_List2[3][j] > (width_m + char_List2[3][j-1])):

            if (j+1) == len(char_List2[3])-1:
                out_modify = out_modify+ ' ' + label_temp[max_index]
                out = out + out_modify + ' '
                out_modify = ""
                i += 1
                continue
                #out_modify = speling.correct(out_modify)
                #out_modify += label_temp[new_d[0][1]]
            out_modify += ' '
                #out = out + out_modify
                #output_text.append(' ')
                #output_text.append(out_modify)
                #print(out_modify)
                #out_modify = ""

            #行の最後の時
        if (j+1) == len(char_List2[3])-1:
            out_modify = out_modify + label_temp[max_index]
            #out_modify = speling.correct(out_modify)
            out = out + out_modify + " "
            out_modify = ""
            i += 1
            continue
            #print(label_temp[new_d[0][1]])
        out_modify = out_modify + label_temp[max_index]
           # print(out_modify)
        i += 1
        continue

    return out
   
def match_text(img_temp,label_temp,frame):
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    window_img = frame
    #sフレームの青い部分を二値化
    blue_threshold_img = cut_blue_img1(window_img)
    #plt.imshow(blue_threshold_img)
    #plt.show()
    #コーナー検出
    try:
        p1,p2,p3,p4 = points_extract1(blue_threshold_img)
    except TypeError:
        print("jj")
        print("Screen cannot be detected")
        return [],[]

    #コーナーに従って画像の切り取り
    cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]]
    #射影変換
    syaei_img = projective_transformation(window_img,p1,p2,p3,p4)
    #対象画像をリサイズ
    syaei_resize_img = cv2.resize(syaei_img,dsize=(610,211))
    #対象画像をグレイスケール化
    gray_img = cv2.cvtColor(syaei_resize_img,cv2.COLOR_BGR2GRAY)
    #二値画像へ
    ret, img_mask = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)
    #img_mask = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,-3)
    #ノイズ除去
    img_mask = cv2.medianBlur(img_mask,3)
    #膨張化
    img_mask = cv2.dilate(img_mask,kernel)
    #高さ、幅を保持
    height,width = img_mask.shape
    #縦方向のProjection Profileを保持
    array_H = Projection_H(img_mask,height,width)
    #縦方向の最大値を保持
    H_THRESH = max(array_H)
    char_List1 = Detect_HeightPosition(H_THRESH,height,array_H)

    #if (len(char_List1) % 2) == 0:
        #print("Screen cannot be detected")
        #return [], []
        
    out_modify = "" #修正したテキスト
    output_text = [] #読み取ったテキスト
    s = {}
    new_d = {}
    out = "" #読み取ったテキスト
    for i in range(0,len(char_List1)-1,2):
        #行ごとに画像を切り取る
        img_h = img_mask[int(char_List1[i]):int(char_List1[i+1]),:]
        height_h , width_h =img_h.shape
        #横方向のProjection Profileを得る
        array_V = Projection_V(img_h,height_h,width_h)
        W_THRESH = max(array_V)
        char_List2 = Detect_WidthPosition(W_THRESH,width_h,array_V)
        for j in range(0,len(char_List2)-1,2):
            #end_time = time.perf_counter()
            #print(end_time-start_time)
            new_d = {}
            s={}
            #一文字ずつ切り取る
            match_img = img_mask[int(char_List1[i])-2:int(char_List1[i+1])+2,int(char_List2[j])-1:int(char_List2[j+1])+1]
            #cv2.imwrite("match.jpg",match_img)
            try:
                match_img = cv2.resize(match_img,dsize=(26,36))
            except cv2.error:
                return [],[]
            height_m,width_m = match_img.shape
            img_g = cv2.rectangle(syaei_resize_img, (int(char_List2[j]) ,int(char_List1[i])), (int(char_List2[j+1]), int(char_List1[i+1])), (0,0,255), 2)
            
            for f in range(len(label_temp)):
                temp_th = img_temp[f]
                temp_th = cv2.resize(temp_th,dsize=(26,36))
                #テンプレートマッチング
                #入力画像、テンプレート画像、類似度の計算方法が引数 返り値は検索窓の各市でのテンプレート画像との類似度を表す二次元配列
                match = cv2.matchTemplate(match_img,temp_th,cv2.TM_CCORR_NORMED)
                en = time.perf_counter()
                #返り値は最小類似点、最大類似点、最小の場所、最大の場所
                min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
                #からのリストに
                
                s.setdefault(max_value,f)
            #print(end-start)
            #類似度が最大のもの順にソート
            new_d = sorted(s.items(), reverse = True)
            #print(label_temp[new_d[0][1]])
            #print(new_d[0][0])
            #print(label_temp[new_d[1][1]])
            #print(new_d[1][0])     
            #new_d[0][1]がlabelの番号、new_d[0][0]が最大類似度
            #print(char_List2)
            #print(width_m)
            #空白があるとき
            if new_d[0][0] < 0.7:
                continue
            if (j != 0) & (char_List2[j] > (width_m + char_List2[j-1])):

                if (j+1) == len(char_List2)-1:
                    out_modify = out_modify+ ' ' + label_temp[new_d[0][1]]
                    out = out + out_modify + ' '
                    out_modify = ""
                    new_d = {}
                    continue
                #out_modify = speling.correct(out_modify)
                #out_modify += label_temp[new_d[0][1]]
                out_modify += ' '
                #out = out + out_modify
                #output_text.append(' ')
                #output_text.append(out_modify)
                #print(out_modify)
                #out_modify = ""
                

            #行の最後の時
            if (j+1) == len(char_List2)-1:
                out_modify = out_modify + label_temp[new_d[0][1]]
                #out_modify = speling.correct(out_modify)
                out = out + out_modify + " "
                out_modify = ""
                new_d = {}
                continue
            #print(label_temp[new_d[0][1]])
            out_modify = out_modify + label_temp[new_d[0][1]]
           # print(out_modify)
            new_d = {}
            continue

    return out
#二次元リストから同じものを削除
def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

def match_text2(img_temp,label_temp,frame):
    #対象画像をリサイズ
    #対象画像をグレイスケール化
    #gray_img = cv2.cvtColor(syaei_resize_img,cv2.COLOR_BGR2GRAY)
    #二値画像へ
    #ret, img_mask = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)
    img_mask = frame
    #img_mask = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,-3)
    #ノイズ除去
    #img_mask = cv2.medianBlur(img_mask,3)
    #膨張化
    #img_mask = cv2.dilate(img_mask,kernel)
    #高さ、幅を保持
    height,width = img_mask.shape
    out_modify = "" #修正したテキスト
    #横方向のProjection Profileを得る
    array_V = Projection_V(img_mask,height,width)
    W_THRESH = max(array_V)
    char_List2 = Detect_WidthPosition(W_THRESH,width,array_V)

    out_modify = "" #修正したテキスト
    out = "" #読み取ったテキスト
    #横方向にきって列ごとに保存
    temp_th = [ cv2.resize(img_temp[i],dsize=(26,36)) for i in range(len(img_temp))]
    #print(len(temp_th))

    for j in range(0,len(char_List2)-1,2):
        #end_time = time.perf_counter()
        #print(end_time-start_time)
        #一文字ずつ切り取る
        match_img = img_mask[:,int(char_List2[j])-1:int(char_List2[j+1])+1]
        #cv2.imwrite("match.jpg",match_img)
        try:
            match_img = cv2.resize(match_img,dsize=(26,36))
        except cv2.error:
            return ""
        height_m,width_m = match_img.shape
        match = [cv2.matchTemplate(match_img,temp_th[i],cv2.TM_CCORR_NORMED) for i in range(len(label_temp))]
        max_value = [cv2.minMaxLoc(match[i])[1] for i in range(len(label_temp))]
        max_index = np.argmax(max_value)
        max_v = max_value[max_index]

        if max_v < 0.6:
            continue
        if (j != 0) & (char_List2[j] > (width_m + char_List2[j-1])):

            if (j+1) == len(char_List2)-1:
                out_modify = out_modify+ ' ' + label_temp[max_index]
                out = out + out_modify + ' '
                #output_text.append('\n')
                out_modify = ""
                continue
                #out_modify = speling.correct(out_modify)
                #out_modify += label_temp[new_d[0][1]]
            out_modify += ' '
                #out = out + out_modify
                #output_text.append(' ')
                #output_text.append(out_modify)
                #print(out_modify)
                #out_modify = ""
                
        #行の最後の時
        if (j+1) == len(char_List2)-1:
            out_modify = out_modify + label_temp[max_index]
            #out_modify = speling.correct(out_modify)
            out = out + out_modify + ' '
            #output_text.append('\n')
            out_modify = ""
            new_d = {}
            continue
        #print(label_temp[new_d[0][1]])
        out_modify = out_modify + label_temp[max_index]
        #print(out_modify)
        continue

    return out



        














    