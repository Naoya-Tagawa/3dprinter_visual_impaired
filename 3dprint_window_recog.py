import sys
#import path
#sys.path.append("C:/Users/Naoya Tagawa/AppData/Local/Programs/Python/Python310/Lib/site-packages")
import cv2
import time
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
import tkinter as tk
import tkinter.ttk as ttk
import threading
from PIL import Image
#import pyttsx3 あとでインストールする

def syaei(img1,p1,p2,p3,p4):
    #座標
    #p1 左上
    #p2 左下
    #p3 右上
    #p4 右下
    #幅
    w = np.linalg.norm(p3[0]-p1[0])
    w = math.floor(w*1.1)
    h = np.linalg.norm(p2[1]-p1[1])
    h = math.floor(h*1.1)
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

def camera(cycle,count):
    cap = cv2.VideoCapture(0)
    read_fps = cap.get(cv2.CAP_PROP_FPS)
    print(read_fps)
    time_counter = 0
    while True:
        ret , frame = cap.read()
        #フレームが取得できない場合は画面を閉じる
        if not ret:
            cv2.destroyAllWindows()
        time_counter += 1
        cv2.imshow("frame",frame)
        #フレームカウントがthreshを超えたら処理
        if(time_counter >= cycle):
            time_counter = 0
            match_text(frame,count)
            count+=1
        #qキーが入力されたら画面を閉じる
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

def match_text(frame,count_w):
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    window_img = frame
    #フレームの青い部分を二値化
    blue_threshold_img = cut_blue_img(window_img)
    #コーナー検出
    p1,p2,p3,p4 = points_extract(blue_threshold_img)
    #コーナーに従って画像の切り取り
    cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]]
    #射影変換
    syaei_img = syaei(window_img,p1,p2,p3,p4)
    #対象画像をリサイズ
    syaei_resize_img = cv2.resize(syaei_img,dsize=(610,211))
    #対象画像をグレイスケール化
    gray_img = cv2.cvtColor(syaei_resize_img,cv2.COLOR_BGR2GRAY)
    #二値画像へ
    #ret, img_mask = cv2.threshold(gray_img,130,255,cv2.THRESH_BINARY)
    img_mask = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,-3)
    #ノイズ除去
    img_mask = cv2.medianBlur(img_mask,3)
    #膨張化
    img_mask = cv2.dilate(img_mask,kernel)
    count1=0
    index = []
    output_text = []
    count=0
    count_first = 0
    s = {}
    new_d = {}
    like = {}
    l = 0
    like_x = {}
    out = ""
    for f in window_z:
        l = 0
        while True:
            s = {}
            new_d = {}
            x, y , w , h = f
            x = x + l
            match_img = img_mask[y:y+h,x:x+w]
            for i in range(len(temp['x'])):
                temp_th = img_temp[i]
                temp_th = cv2.resize(temp_th,dsize = (26,36))
                #テンプレートマッチング
                #入力画像、テンプレート画像、類似度の計算方法が引数 返り値は検索窓の各市でのテンプレート画像との類似度を表す二次元配列
                match = cv2.matchTemplate(match_img,temp_th,cv2.TM_CCORR_NORMED)
                #返り値は最小類似点、最大類似点、最小の場所、最大の場所
                min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
                #からのリストに
                s.setdefault(max_value,i)
                #類似度が最大のもの順にソート
            new_d = sorted(s.items(), reverse = True)
            #類似度が最大のものを辞書に格納
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
            elif l== 5:
                l = -1
            elif l == -1:
                l = -2
            elif l == -2:
                break
        l = 0
        #左右に動かして一番類似度が大きいものをmid_vに格納
        mid_v = sorted(like_x.items(),reverse = True)
        #x1に最も類似度が大きい位置のx座標を格納
        x1 = mid_v[0][1]
        like_x = {}
        
        #上下に動かす
        while True:
            s = {}
            new_d = {}
            x, y , w , h = f
            x = x1
            y = y + l
            match_img = img_mask[y:y+h,x:x+w]
            for i in range(len(temp['x'])):
                temp_th = img_temp[i]
                temp_th = cv2.resize(temp_th,dsize = (26,36))
                #テンプレートマッチング
                #入力画像、テンプレート画像、類似度の計算方法が引数 返り値は検索窓の各市でのテンプレート画像との類似度を表す二次元配列
                match = cv2.matchTemplate(match_img,temp_th,cv2.TM_CCORR_NORMED)
                #返り値は最小類似点、最大類似点、最小の場所、最大の場所
                min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
                #からのリストに
                s.setdefault(max_value,i)
                #類似度が最大のもの順にソート
            new_d = sorted(s.items(), reverse = True)
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
        if max_v[0][0] < 0.7:
            out = out + ' '
            output_text.append(' ')
            like = {}
        
            if x == 558:
                #print(label_temp[max_v[0][1]])
                output_text.append('\n')
                out = out + "\n"
            continue
        
        if x == 558:
            #print(label_temp[max_v[0][1]])
            output_text.append(label_temp[max_v[0][1]])
            out = out + label_temp[max_v[0][1]] + "\n"
            like = {}
            continue
        output_text.append(label_temp[max_v[0][1]])
        out = out + label_temp[max_v[0][1]]
        like = {}
        continue
    
    file_w(out,output_text)
    cv2.imwrite(r'C:\Users\Naoya Tagawa\Desktop\answer\window{0}.jpg'.format(count_w),syaei_img)
    plt.imshow(img_mask)
    plt.show()

def file_w(text,output_text):
    f = open('3dprint_window.txt',mode='a',encoding = 'UTF-8')
    f.write(text)
    f.write('\n')
    f.close()
        
    #engine = pyttsx3.init()
    #rate = engine.getProperty("rate")
    #engine.getProperty("rate")
    #engine.setProperty("rate",200)
    #engine.setProperty("voice","english")
    #volume = engine.getProperty('volume')
    #engine.setProperty('volume',1.0)
    #print(output_text)
    #for word in output_text:
        #if '\n' in word:
            #word = "".join(word)
            #engine.say(word)
            #time.sleep(3)
            #continue
        #else:
            #engine.say(word)
            
    #if text !="not":
        #engine.say("現在カーソルが示しているのは")
        #text = "".join(text)
        #engine.say(text)
        #engine.say("です")
    #engine.runAndWait()
    
#テンプレートをロード
temp = np.load(r'C:\Users\Naoya Tagawa\OneDrive\dataset.npz')
#テンプレート画像を格納
img_temp = temp['x']
#テンプレートのラベル(文)を格納
label_temp = temp['y']
count=0
window_z = ((21, 18, 26, 36),(50, 18, 25, 36),(78, 18, 26, 36),(106, 18, 26, 36),(134, 18, 26, 36),(162,18,26,36),(190,18,26,36),(218,18,26,36),(246,18,26,36),(274,18,26,36),(302,18,26,36),(332,18,26,36),(362,18,26,36),(390,18,26,36),(418,18,26,36),(448,18,26,36) , (476,18,26,36),(502,18,26,36),(530,18,26,36),(558, 18, 26, 36), 
(21,63,26,36),(50, 63, 26, 36), (78, 63, 26, 36), (106, 63, 26, 36), (134, 63, 26, 36), (162, 63, 26, 36), (190, 63, 26, 36), (218, 63, 26, 36), (246, 63, 26, 36), (274, 63, 26, 36), (302, 63, 26, 36), (332, 63, 26, 36), (362, 63, 26, 36), (390, 63, 26, 36), (418, 63, 26, 37), (448, 63, 26, 36), (476, 63, 26, 36), (502, 63, 26, 36), (530, 63, 26, 36), (558, 63, 26, 36),
(21,107,26,36), (50, 107, 26, 36), (78, 107, 26, 36), (106, 107, 26, 36), (134, 107, 26, 36),(162, 107, 26, 36), (190, 107, 26, 36), (218, 107, 26, 36), (246, 107, 26, 36), (274, 107, 26, 36), (302, 107, 26, 36), (332, 107, 26, 36), (362, 107, 26, 36), (390, 107, 26, 36), (418, 107, 26, 36), (448, 107, 26, 36), (476, 107, 26, 36),(502,107,26,37),(530, 107, 26, 36),(558, 107, 26, 36), 
(21,150,26,36),(50, 150, 26, 36), (78, 150, 26, 36), (106, 150, 26, 36), (134, 150, 26, 36), (162, 150, 26, 36), (190, 150, 26, 36), (218, 150, 26, 36), (246, 150, 26, 36), (274, 150, 26, 36), (302, 150, 26, 36), (332, 150, 26, 36), (362, 151, 26, 36), (390, 151, 26, 36), (418, 151, 26, 36), (448, 151, 26, 36), (476, 151, 26, 36), (502, 151, 26, 36), (530, 151, 26, 36), (558, 151, 26, 36))
camera(300,count)