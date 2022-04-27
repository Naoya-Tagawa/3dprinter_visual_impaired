import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
from PIL import Image
import difflib
from dictionary_word import speling
import pyttsx3
import time

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

#対象画像をロード
img = cv2.imread(r".\camera1\camera25.jpg")
c_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
m_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(c_img)
plt.show()
#青い部分のみを二値化
close_img = cut_blue_img(img)
#コーナー検出
p1,p2,p3,p4 = points_extract(close_img)
#コーナーに従って画像の切り取り
img_k = img[p1[1]:p2[1],p2[0]:p3[0]]
#射影変換
syaei_img = syaei(img,p1,p2,p3,p4)
#cv2.imwrite(r'C:\Users\Naoya Tagawa\OneDrive\s.jpg',syaei_img)
s_img = cv2.cvtColor(syaei_img,cv2.COLOR_BGR2RGB)
plt.imshow(s_img)
plt.show()

#テンプレートをロード
temp = np.load(r"./dataset.npz")
#テンプレート画像を格納
img_temp = temp['x']
#テンプレートのラベル(文)を格納
label_temp = temp['y']
#カーネル
kernel = np.ones((3,3),np.uint8)
#対象画像をリサイズ
syaei_resize_img = cv2.resize(syaei_img,dsize=(610,211)) 
#対象画像をグレイスケール化
gray_img = cv2.cvtColor(syaei_resize_img,cv2.COLOR_BGR2GRAY)
#二値画像へ
#ret,img_mask = cv2.threshold(gray_img,130,255,cv2.THRESH_BINARY)
img_mask = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,-3)
#ノイズ除去
img_mask = cv2.medianBlur(img_mask,3)
#膨張化
img_mask = cv2.dilate(img_mask,kernel)

#画像の確認
plt.imshow(img_mask,cmap='gray')
plt.show()

window_z = ((21, 18, 26, 36),(50, 18, 25, 36),(78, 18, 26, 36),(106, 18, 26, 36),(134, 18, 26, 36),(162,18,26,36),(190,18,26,36),(218,18,26,36),(246,18,26,36),(274,18,26,36),(302,18,26,36),(332,18,26,36),(362,18,26,36),(390,18,26,36),(418,18,26,36),(448,18,26,36) , (476,18,26,36),(502,18,26,36),(530,18,26,36),(558, 18, 26, 36), 
(21,63,26,36),(50, 63, 26, 36), (78, 63, 26, 36), (106, 63, 26, 36), (134, 63, 26, 36), (162, 63, 26, 36), (190, 63, 26, 36), (218, 63, 26, 36), (246, 63, 26, 36), (274, 63, 26, 36), (302, 63, 26, 36), (332, 63, 26, 36), (362, 63, 26, 36), (390, 63, 26, 36), (418, 63, 26, 37), (448, 63, 26, 36), (476, 63, 26, 36), (502, 63, 26, 36), (530, 63, 26, 36), (558, 63, 26, 36),
(21,107,26,36), (50, 107, 26, 36), (78, 107, 26, 36), (106, 107, 26, 36), (134, 107, 26, 36),(162, 107, 26, 36), (190, 107, 26, 36), (218, 107, 26, 36), (246, 107, 26, 36), (274, 107, 26, 36), (302, 107, 26, 36), (332, 107, 26, 36), (362, 107, 26, 36), (390, 107, 26, 36), (418, 107, 26, 36), (448, 107, 26, 36), (476, 107, 26, 36),(502,107,26,37),(530, 107, 26, 36),(558, 107, 26, 36), 
(21,150,26,36),(50, 150, 26, 36), (78, 150, 26, 36), (106, 150, 26, 36), (134, 150, 26, 36), (162, 150, 26, 36), (190, 150, 26, 36), (218, 150, 26, 36), (246, 150, 26, 36), (274, 150, 26, 36), (302, 150, 26, 36), (332, 150, 26, 36), (362, 151, 26, 36), (390, 151, 26, 36), (418, 151, 26, 36), (448, 151, 26, 36), (476, 151, 26, 36), (502, 151, 26, 36), (530, 151, 26, 36), (558, 151, 26, 36))

count1=0
head = 0
out_modify = ""
index = []
output_text = []
before_text = []
count=0
count_first = 0
s = {}
new_d = {}
like = {}
l = 0
like_x = {}
out = ""
before_kersol =""
kersol = ""
before_text = ""
for f in window_z:
    l = 0
    while True:
        s = {}
        new_d = {}
        x, y , w , h = f
        x = x + l
        match_img = img_mask[y:y+h,x:x+w]
        #plt.imshow(match_img)
        #plt.show()
        for i in range(len(temp['x'])):
            temp_th = img_temp[i]
            #plt.imshow(temp_th)
            #plt.show()
            #print(match_img.shape)
            temp_th = cv2.resize(temp_th,dsize = (26,36))
            #テンプレートマッチング
            #入力画像、テンプレート画像、類似度の計算方法が引数 返り値は検索窓の各市でのテンプレート画像との類似度を表す二次元配列
            match = cv2.matchTemplate(match_img,temp_th,cv2.TM_CCORR_NORMED)
            #返り値は最小類似点、最大類似点、最小の場所、最大の場所
            min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
            # ptに類似度が最大(値が最小だから)のmin_valueの場所min_ptを格納
            pt = max_pt
            #からのリストに
            s.setdefault(max_value,i)
            #類似度が最大のもの順にソート
        new_d = sorted(s.items(), reverse = True)
        #print(label_temp[new_d[0][1]])
        #print(new_d[0][0])
        like.setdefault(new_d[0][0],new_d[0][1])
        like_x.setdefault(new_d[0][0],x)
        if l == 0:
            l = 1
        elif l == 1:
            l = 2
        elif l == 2:
            l = 3
        elif l == 3:
            l = 4
        elif l == 4:
            l = 5
        elif l == 5:
            l = -1
        elif l == -1:
            l = -2
        elif l == -2:
            break
    l = 0
    mid_v = sorted(like_x.items(),reverse = True)      
    x1 = mid_v[0][1]
    like_x = {}
    while True:
        s = {}
        new_d = {}
        x, y , w , h = f
        x = x1
        y = y + l
        match_img = img_mask[y:y+h,x:x+w]
        #ma = cv2.cvtColor(match_img,cv2.COLOR_BGR2RGB)
        #plt.imshow(ma)
        #plt.show()
        for i in range(len(temp['x'])):
            temp_th = img_temp[i]
            #plt.imshow(temp_th)
            #plt.show()
            #print(match_img.shape)
            temp_th = cv2.resize(temp_th,dsize = (26,36))
            #テンプレートマッチング
            #入力画像、テンプレート画像、類似度の計算方法が引数 返り値は検索窓の各市でのテンプレート画像との類似度を表す二次元配列
            match = cv2.matchTemplate(match_img,temp_th,cv2.TM_CCORR_NORMED)
            #返り値は最小類似点、最大類似点、最小の場所、最大の場所
            min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
            # ptに類似度が最大(値が最小だから)のmin_valueの場所min_ptを格納
            pt = max_pt
            #からのリストに
            s.setdefault(max_value,i)
            #類似度が最大のもの順にソート
        new_d = sorted(s.items(), reverse = True)
        #print(label_temp[new_d[0][1]])
        like.setdefault(new_d[0][0],new_d[0][1])
        if l == 0:
            l = 1
        elif l == 1:
            l = 2
        elif l == 2:
            l = -1
        elif l == -1:
            l = -2
        elif l == -2:
            break
    x , y , w , h = f
    
    max_v = sorted(like.items(),reverse = True)
    #print(max_v)
    #print(label_temp[max_v[0][1]])
    if max_v[0][0] < 0.7:
        
        if head == 0:
            out = out + ' '
            output_text.append(' ')
            head = 1
            like = {}
            
            if x == 558:

                out_modify = speling.correct(out_modify)
                out = out + out_modify + "\n"
                output_text.append(out_modify)
                output_text.append("\n")
                head = 0
                out_modify = ""
                like = {}
                continue
                
            continue
            
            
        if x == 558:
            
            out_modify = speling.correct(out_modify)
            out_modify = out_modify + ' '
            out = out + out_modify + "\n"
            output_text.append(out_modify)
            output_text.append('\n')
            head = 0
            out_modify = ""
            like = {}
            continue
            
        print(out_modify)
        out_modify = speling.correct(out_modify)
        print(out_modify)
        out_modify = out_modify + ' '
        out = out + out_modify
        output_text.append(out_modify)
        like = {}
        out_modify = ""
        continue
        
    if head == 0:
        out_modify = out_modify + label_temp[max_v[0][1]]
        head = 1
        like = {}
    
        if x == 558:
            out_modify = speling.correct(out_modify)
            out_modify = out_modify + "\n"
            out = out + out_modify
            output_text.append(out_modify)
            like = {}
            out_modify = ""
            head = 0
            continue
            
        continue
        
        
    if x == 558:
        out_modify = out_modify + label_temp[max_v[0][1]]
        out_modify = speling.correct(out_modify)
        output_text.append(out_modify)
        output_text.append("\n")
        out = out + out_modify + "\n"
        like = {}
        out_modify = ""
        head = 0
        continue
        
    out_modify = out_modify + label_temp[max_v[0][1]]
    #print(out_modify)
    like = {}
    continue
    
def kersol_search(text):
    i = 0
    kersol = ""
    for word in text:
        if word[0] == ">":
            i = 1
            kersol += word
        elif i == 1:
            kersol += word
        if (i == 1) & (word == '\n'):
            i = 0
    return kersol 

ikersol = kersol_search(output_text)
print(out)
print(output_text)
print(ikersol)
def text_read(text):
    engine = pyttsx3.init()
    #rateはデフォルトが200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',150)
    #volume デフォルトは1.0 設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',1.0)
    for word in text:
        if word == '\n':
            time.sleep(5)
        engine.say(word)
        engine.runAndWait()

text_read(output_text)
