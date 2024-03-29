import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import difflib
import time
import cv2
import numpy as np
import glob
from natsort import natsorted
import threading
from PIL import Image , ImageTk , ImageOps
import pyttsx3 
from dictionary_word import speling
import difflib
import numpy as np
import cv2
import matplotlib.pyplot as plt

#話すスピード
speed = 150
#ボリューム
vol = 1.0

global count
global before_text
global before_kersol
global before_window_img
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
    print(M)
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

def camera():
    #cap = cv2.VideoCapture(1)
    #read_fps = cap.get(cv2.CAP_PROP_FPS)
    #print(read_fps)
    #画面が遷移したか調査
    diff_match_text(img1,img2)


def diff_match_text(before_frame,present_frame):
    output_text = []
    out = ""
    out_modify = ""
    s = {}
    new_d = {}
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    #フレームの青い部分を二値化
    blue_threshold_before_img = cut_blue_img(before_frame)
    blue_threshold_present_img = cut_blue_img(present_frame)
    #コーナー検出
    try:
        before_p1,before_p2,before_p3,before_p4 = points_extract(blue_threshold_before_img)
        present_p1,present_p2,present_p3,present_p4 = points_extract(blue_threshold_present_img)
    except TypeError:
        print("Screen cannot be detected")
        return before_window_img,[] ,[]

    #コーナーに従って画像の切り取り
    #cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]]
    cut_present = present_frame[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    cut_before = before_frame[before_p1[1]:before_p2[1],before_p2[0]:before_p3[0]]
    #射影変換
    #syaei_before_img = syaei(before_frame,before_p1,before_p2,before_p3,before_p4)
    syaei_present_img = syaei(present_frame,present_p1,present_p2,present_p3,present_p4)
    cv2.imshow("ss",syaei_present_img)
    cv2.waitKey(0)
    #対象画像をリサイズ
    syaei_resize_before_img = cv2.resize(cut_before,dsize=(610,211))
    syaei_resize_present_img = cv2.resize(cut_present,dsize=(610,211))
    copy =present_frame
    #plt.imshow(syaei_resize_present_img)
    #plt.show()
    #frame_diff = cv2.absdiff(syaei_resize_present_img,syaei_resize_before_img)
    frame_diff = cv2.absdiff(present_frame,before_frame)
    #グレイスケール化
    gray_frame_diff = cv2.cvtColor(frame_diff,cv2.COLOR_BGR2GRAY)
    #ノイズ除去
    gray_frame_diff = cv2.medianBlur(gray_frame_diff,3)
    #二値画像へ
    ret, mask_frame_diff = cv2.threshold(gray_frame_diff,0,255,cv2.THRESH_OTSU)
    frame_diff = np.where(syaei_resize_present_img >= syaei_resize_before_img, syaei_resize_present_img-syaei_resize_before_img,0)
    #ret, mask_frame_diff = cv2.threshold(gray_frame_diff,32,255,cv2.THRESH_BINARY)
    #gray_frame = cv2.cvtColor(frame_diff,cv2.COLOR_BGR2GRAY)
    frame_diff = (frame_diff > 32) *255
    #cv2.imwrite("frame_diff2.jpg",frame_diff)
    #mask_frame_diff = cv2.dilate(mask_frame_diff,kernel)
    cv2.imwrite("frame_diff3.jpg",mask_frame_diff)
    #コーナーに従って画像の切り取り
    #cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]
    cut_present = present_frame[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    cut_before = before_frame[before_p1[1]:before_p2[1],before_p2[0]:before_p3[0]]
    #射影変換
    #syaei_before_img = syaei(before_frame,before_p1,before_p2,before_p3,before_p4)
    #syaei_present_img = syaei(present_frame,present_p1,present_p2,present_p3,present_p4)
    #対象画像をリサイズ

    syaei_resize_before_img = cv2.resize(cut_before,dsize=(610,211))
    syaei_resize_present_img = cv2.resize(cut_present,dsize=(610,211))
    copy =syaei_resize_present_img
    #plt.imshow(syaei_resize_present_img)
    #plt.show()
    frame_diff = cv2.absdiff(syaei_resize_present_img,syaei_resize_before_img)
    #frame_diff = np.where(syaei_resize_present_img >= syaei_resize_before_img, syaei_resize_present_img-syaei_resize_before_img,0)
    #グレイスケール化
    gray_frame_diff = cv2.cvtColor(frame_diff,cv2.COLOR_BGR2GRAY)
    #ノイズ除去
    gray_frame_diff = cv2.medianBlur(gray_frame_diff,3)
    #二値画像へ
    ret, mask_frame_diff = cv2.threshold(gray_frame_diff,0,255,cv2.THRESH_OTSU)
    #ret, mask_frame_diff = cv2.threshold(gray_frame_diff,32,255,cv2.THRESH_BINARY)
    #gray_frame = cv2.cvtColor(frame_diff,cv2.COLOR_BGR2GRAY)
    frame_diff = (frame_diff > 10) *255
    cv2.imwrite("frame_diff.jpg",frame_diff)
    #mask_frame_diff = cv2.dilate(mask_frame_diff,kernel)
    cv2.imwrite("frame_diff1.jpg",mask_frame_diff)


    #グレイスケール化
    #gray_frame_before_diff = cv2.cvtColor(syaei_resize_before_img,cv2.COLOR_BGR2GRAY)
    #二値画像へ
    #ret, img_before_mask = cv2.threshold(gray_frame_before_diff,0,255,cv2.THRESH_OTSU)
    #ノイズ除去
    #img_before_mask = cv2.medianBlur(img_before_mask,3)
    #膨張化
    #img_before_mask = cv2.dilate(img_before_mask,kernel)
    #グレイスケール化
    #gray_frame_present_diff = cv2.cvtColor(syaei_resize_present_img,cv2.COLOR_BGR2GRAY)
    #二値画像へ
    #ret, #img_present_mask = cv2.threshold(gray_frame_present_diff,0,255,cv2.THRESH_OTSU)
    #ノイズ除去
    #img_present_mask = cv2.medianBlur(#img_present_mask,3)
    #膨張化
    #img_present_mask = cv2.dilate(#img_present_mask,kernel)

    #frame_diff = cv2.absdiff(img_before_mask,#img_present_mask)
    #frame_diff = (frame_diff > 32) *255
    #cv2.imwrite("frame_diff.jpg",frame_diff)
    #gray_frame = cv2.cvtColor(frame_diff,cv2.COLOR_BGR2GRAY)
    height , width = mask_frame_diff.shape
    array_H = Projection_H(mask_frame_diff,height,width)
    H_THRESH = max(array_H)
    char_List1 = Detect_HeightPosition(H_THRESH,height,array_H)
    for i in range(0,len(char_List1)-1,2):
        img_h = mask_frame_diff[int(char_List1[i]):int(char_List1[i+1]),:]
        img_j = cv2.rectangle(syaei_resize_present_img, (0 ,int(char_List1[i])), (610, int(char_List1[i+1])), (0,0,255), 2)
        height_h , width_h =img_h.shape
        #横方向のProjection Profileを得る
        array_V = Projection_V(img_h,height_h,width_h)
        W_THRESH = max(array_V)
        char_List2 = Detect_WidthPosition(W_THRESH,width_h,array_V)
        for j in range(0,len(char_List2)-1,2):
            #一文字ずつ切り取る
            #img_f = cv2.rectangle(syaei_resize_present_img, (int(char_List2[j]) ,int(char_List1[i])), (int(char_List2[j+1]), int(char_List1[i+1])), (0,0,255), 2)
            print("k")
    #cv2.imwrite("difference2.png",img_f)
    cv2.imwrite("diffecence3.jpg",img_j)

def match_text(frame):
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    window_img = frame
    #フレームの青い部分を二値化
    blue_threshold_img = cut_blue_img(window_img)
    #コーナー検出
    try:
        p1,p2,p3,p4 = points_extract(blue_threshold_img)
    except TypeError:
        print("Screen cannot be detected")
        return before_window_img,[] ,[]

    #コーナーに従って画像の切り取り
    cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]]
    #射影変換
    syaei_img = syaei(window_img,p1,p2,p3,p4)
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
            match_img = cv2.resize(match_img,dsize=(26,36))
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
                    out = out + out_modify + "\n"
                    output_text.append(out_modify)
                    output_text.append('\n')
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
                out = out + out_modify + "\n"
                output_text.append(out_modify)
                output_text.append('\n')
                out_modify = ""
                new_d = {}
                continue
            #print(label_temp[new_d[0][1]])
            out_modify = out_modify + label_temp[new_d[0][1]]
           # print(out_modify)
            new_d = {}
            continue

    print(output_text)
    print(out)
    cv2.imwrite("difference1.jpg",img_g)
    return img_mask , output_text, out 

def voice(img_likely_ratio,output_text,out):
    #現在のカーソル
    present_kersol = kersol_search(output_text)
    before = []
    after = []
    #前と後のカーソルの類似度
    s = difflib.SequenceMatcher(None,before_kersol,present_kersol)
    #print(s.ratio())
    if kersol_exist_search(before_kersol,out) == True: #前のカーソルがある(全画面変わっていない)
        if s.ratio() <= 0.50: #カーソルが変わっていたら
            engine = pyttsx3.init()
            #rateはデフォルトが200
            voice = engine.getProperty('voices')
            engine.setProperty("voice",voice[1].id)
            rate = engine.getProperty('rate')
            engine.setProperty('rate',speed)
            #volume デフォルトは1.0 設定は0.0~1.0
            volume = engine.getProperty('volume')
            engine.setProperty('volume',vol)
            engine.say("cursor was changed from")
            partial_text_read(before_kersol)
            engine.say("to")
            partial_text_read(present_kersol)
            engine.runAndWait()
    
    elif (len(before_kersol) == 0) & (len(present_kersol) == 0): #前のカーソルも今のカーソルもない(数値の画面が変わった)
    #類似度90%は変化部分を読む
        res = difflib.ndiff(before_text,output_text)
        for word in res:
            if (word[0] == '-'):
                before.append(word[2:])
            elif word[0] == '+':
                after.append(word[2:])
        if (0< len(before) < 6) & (0 < len(after) < 6):
            engine = pyttsx3.init()
            voice = engine.getProperty('voices')
            engine.setProperty("voice",voice[1].id)
            #rateはデフォルトが200
            rate = engine.getProperty('rate')
            engine.setProperty('rate',speed)
            #volume デフォルトは1.0 設定は0.0~1.0
            volume = engine.getProperty('volume')
            engine.setProperty('volume',vol)
            engine.say("Changed from")
            whole_text_read(before)
            engine.say("to")
            whole_text_read(after)
            engine.runAndWait()
        else:
            whole_text_read(output_text)
            
    else: #全画面変化
        whole_text_read(output_text)
        engine = pyttsx3.init()
        voice = engine.getProperty('voices')
        engine.setProperty("voice",voice[1].id)
        rate = engine.getProperty('rate')
        engine.setProperty('rate',speed)
        #volume デフォルトは1.0 設定は0.0~1.0
        volume = engine.getProperty('volume')
        engine.setProperty('volume',vol)
        if kersol_exist_search == True:
            engine.say("The current cursor position is")
            partial_text_read(present_kersol)
            engine.runAndWait()

    #前のテキストを保持
    print(present_kersol)
    before_text = output_text
    before_kersol = present_kersol
    file_w(out,output_text)

#カーソルの表示を探す
def kersol_search(text):
    i = 0
    kersol1 = ""
    for word in text:
        if word[0] == ">":
            i = 1
            kersol1 += word + ' '
        elif (i == 1) & (word == '\n'):
            i = 0
        elif i == 1:
            kersol1 += word
    return kersol1

#カーソルの位置をいう
def kersol_read(text):
    engine = pyttsx3.init()
    voice = engine.getProperty('voices')
    engine.setProperty("voice",voice[1].id)
    #rateはデフォルトが200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',speed)
    #volume デフォルトは1.0 設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',vol)
    count = 0
    engine.say("The current cursor position is")
    for word in text:
        if word == ' ':
            continue
        if '/' in word:
            target = '/'
            idx = word.find(target)
            r = word[:idx]
            engine.say(r)
            engine.say("スラッシュ")
            r = word[idx:]
            engine.say(r)
            continue
        if ',' in word:
            target = ','
            idx = word.find(target)
            r = word[:idx]
            engine.say(r)
            engine.say("カンマ")
            r = word[idx:]
            engine.say(r)
            continue
        if "." in word:
            target = '.'
            idx = word.find(target)
            r = word[:idx]
            engine.say(r)
            engine.say("ドット")
            r = word[idx:]
            engine.say(word)
            continue
        engine.say(word)
    
    engine.runAndWait()

#テキスト全部読み上げ
def whole_text_read(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty("voice",voices[1].id)
    #rateはデフォルトが200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',speed)
    #volume デフォルトは1.0 設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',vol)
    count = 0
    for word in text:
        if word == ' ':
            continue
        if '/' in word:
            target = '/'
            idx = word.find(target)
            r = word[:idx]
            engine.say(r)
            engine.say("スラッシュ")
            r = word[idx:]
            engine.say(r)
            continue
        if ',' in word:
            target = ','
            idx = word.find(target)
            r = word[:idx]
            engine.say(r)
            engine.say("カンマ")
            r = word[idx:]
            engine.say(r)
            continue
        if "." in word:
            target = '.'
            idx = word.find(target)
            r = word[:idx]
            engine.say(r)
            engine.say("ドット")
            r = word[idx:]
            engine.say(r)
            continue
        engine.say(word)
    
    engine.runAndWait()

#カーソルがテキストにあるか
def kersol_exist_search(kersol,text):
    text = text.splitlines()
    for word in text:
        s = difflib.SequenceMatcher(None,kersol[1:],word)
        if s.ratio() >= 0.90:
            return True
    return False

def partial_text_read(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty("voice",voices[1].id)
    #rateはデフォルトが200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',speed)
    #volume デフォルトは1.0 設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',vol)
    engine.say(text)
    engine.runAndWait()



def file_w(text,output_text):
    f = open('3dprint_window.txt',mode='a',encoding = 'UTF-8')
    f.write(text)
    f.write('\n')
    f.close()
        

if __name__ == "__main__":
    #対象画像をロード
    img1 = cv2.imread("./camera1/camera60.jpg")
    img2 = cv2.imread("./camera1/camera61.jpg")
    #テンプレートをロード
    temp = np.load(r'./dataset2.npz')
    #テンプレート画像を格納
    img_temp = temp['x']
    #テンプレートのラベル(文)を格納
    label_temp = temp['y']
    before_text = "Main   →"
    kersol = ">Main"
    count = 0
    print(count)
    #camera_thread = threading.Thread(target = camera)
    #match_thread = threading.Thread(target = match_text)
    #camera_thread.start()
    #match_text(img,before_text,kersol)
    camera()
    match_text(img2)