import multiprocessing
from re import subn
from cv2 import imwrite
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
import multiprocessing
from PIL import Image , ImageTk , ImageOps
import pyttsx3 
from dictionary_word import speling
import difflib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import image_processing
import audio_output
from sklearn.neighbors import NearestNeighbors 
from io import BytesIO
#flag = True: 音声出力
#flag = false: 音声出力しない
import itertools
#話すスピード
speed = 150
#ボリューム
vol = 1.0

global present_img
global before_frame 

def camera():
    global before_frame
    cap = cv2.VideoCapture(1)
    read_fps = cap.get(cv2.CAP_PROP_FPS)
    print(read_fps)
    while True:
        ret , frame = cap.read()
        #フレームが取得できない場合は画面を閉じる
        if not ret:
            cv2.destroyAllWindows()
        cv2.imshow("frame",frame)
        #画面が遷移したか調査
        diff_image_search(before_frame,frame)

        before_frame = frame
        #qキーが入力されたら画面を閉じる
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
        time.sleep(1)

def diff_image_search1(present_frame,before_frame,img_temp,label_temp):
    global present_img
    img = cv2.imread("./black_img.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(img.shape)
    h,w,d = before_frame.shape
    black_window = np.zeros((h,w))
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    #フレームの青い部分を二値化
    blue_threshold_before_img = image_processing.cut_blue_img1(before_frame)
    plt.imshow(blue_threshold_before_img)
    plt.show()
    blue_threshold_present_img = image_processing.cut_blue_img1(present_frame)
    cv2.imwrite("blue_threshold_present.jpg",blue_threshold_present_img)
    plt.imshow(blue_threshold_present_img)
    plt.show()
    #コーナー検出
    #try:
    before_p1,before_p2,before_p3,before_p4 = image_processing.points_extract1(blue_threshold_before_img,before_frame)
    present_p1,present_p2,present_p3,present_p4 = image_processing.points_extract1(blue_threshold_present_img,present_frame)
    #except TypeError:
        #print("Screen cannot be detected")
        #return before_frame,[] ,[]
    #before_frame = cv2.resize(before_frame,dsize=(610,211))
    #present_frame = cv2.resize(present_frame,dsize=(610,211))
    #コーナーに従って画像の切り取り
    cut_present = present_frame[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    cut_before = before_frame[before_p1[1]:before_p2[1],before_p2[0]:before_p3[0]]
    #射影変換
    syaei_before_img = image_processing.projective_transformation(before_frame,before_p1,before_p2,before_p3,before_p4)
    syaei_present_img = image_processing.projective_transformation(present_frame,present_p1,present_p2,present_p3,present_p4)
    cv2.imwrite("syaei_present_frame.jpg",syaei_present_img)
    output_text,out = image_processing.match_text(img_temp,label_temp,present_frame)
    print(out)
    file_w(out,output_text)
    #対象画像をリサイズ
    #syaei_before_img = cv2.resize(syaei_before_img,dsize=(610,211))
    #syaei_present_img = cv2.resize(syaei_present_img,dsize=(610,211))
    #plt.imshow(syaei_present_img)
    #plt.show()
    #plt.imshow(syaei_before_img)
    #plt.show()
    #syaei_resize_before_img = cv2.resize(cut_before,dsize=(610,211))
    #syaei_resize_present_img = cv2.resize(syaei_present_img,dsize=(610,211))
    gray_present_img = cv2.cvtColor(syaei_present_img,cv2.COLOR_BGR2GRAY)
    gray_present_img = cv2.medianBlur(gray_present_img,3)
    ret, mask_present_img = cv2.threshold(gray_present_img,0,255,cv2.THRESH_OTSU)
    #膨張処理
    mask_present_img = cv2.dilate(mask_present_img,kernel)
    height_present,width_present = mask_present_img.shape
    array_present_H = image_processing.Projection_H(mask_present_img,height_present,width_present)
    presentH_THRESH = max(array_present_H)
    present_char_List = image_processing.Detect_HeightPosition(presentH_THRESH,height_present,array_present_H)
    print(present_char_List)
    present_char_List = np.reshape(present_char_List,[int(len(present_char_List)/2),2])
    #present_char_List = image_processing.convert_1d_to_2d(present_char_List,2)
    print(present_char_List)

    #plt.imshow(syaei_resize_present_img)
    #plt.show()
    frame_diff = cv2.absdiff(blue_threshold_present_img,blue_threshold_before_img)
    #frame_diff = cv2.absdiff(present_frame,before_frame)
    #グレイスケール化
    gray_frame_diff = cv2.cvtColor(frame_diff,cv2.COLOR_BGR2GRAY)
    #ノイズ除去
    gray_frame_diff = cv2.medianBlur(gray_frame_diff,3)
    #二値画像へ
    ret, mask_frame_diff = cv2.threshold(gray_frame_diff,0,255,cv2.THRESH_OTSU)
    #膨張処理
    mask_frame_diff = cv2.dilate(mask_frame_diff,kernel)
    cv2.imwrite("frame_diff3.jpg",mask_frame_diff)
    #コーナーに従って画像の切り取り
    #cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]
    mask_cut_diff_frame = mask_frame_diff[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    cv2.imwrite("frame_diff4.jpg",mask_cut_diff_frame)
    height , width = mask_cut_diff_frame.shape
    array_H = image_processing.Projection_H(mask_cut_diff_frame,height,width)
    H_THRESH = max(array_H)
    char_List1 = image_processing.Detect_HeightPosition(H_THRESH,height,array_H)
    print(char_List1)
    #char_List1 = image_processing.convert_1d_to_2d(char_List1,2)
    char_List1 = np.reshape(char_List1,[int(len(char_List1)/2),2])
    print(char_List1)
    #knn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(present_char_List) 
    #distances, indices = knn_model.kneighbors(char_List1)
    #print(indices)
    #indices = image_processing.get_unique_list(indices)
    #print(indices)
    #for  i  in indices:
        #print(i[0])
        #img_j = cv2.rectangle(syaei_present_img, (0,int(present_char_List[i[0]][0])), (610, int(present_char_List[i[0]][1])), (0,0,255), 2)
    #cv2.imwrite("diffecence3.jpg",img_j)
    #文字のみのマスク画像生成
    present_char_List , mask_present_img2 = image_processing.mask_make(blue_threshold_present_img)
    
    engine = pyttsx3.init()
    before_frame_row = []


    for i in present_char_List:
        normal = mask_present_img2.copy()
        cut_present = mask_present_img2[int(i[0]):int(i[1]),]
        cv2.rectangle(normal,(0,0),(w-1,int(i[0])-1),(0,0,0),-1)
        cv2.rectangle(normal,(0,int(i[1])-1),(w-1,h-1),(0,0,0),-1)
        #plt.imshow(normal)
        #plt.show()
        #cut_present_img = syaei_present_img[int(i[0]):int(i[1]),]
        #before_frame_row.append(normal)
        before_frame_row.append(cut_present)
    cv2.imwrite("cut_pre.jpg",before_frame_row[1])
    print(len(before_frame_row))
    if len(present_char_List) == 0:
        return img,img,img,img
    elif len(present_char_List) == 1:
        return before_frame_row[0] , img,img,img
    elif len(present_char_List) == 2:
        return before_frame_row[0] , before_frame_row[1] ,img,img
    elif len(present_char_List) == 3:
        return before_frame_row[0] , before_frame_row[1] ,before_frame_row[2] ,img
    elif len(present_char_List) == 4:
        return before_frame_row[0] , before_frame_row[1],before_frame_row[2],before_frame_row[3]    
    
    
def diff_image_search(before_frame,present_frame,img_temp,label_temp,before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4):
    global present_img
    img = cv2.imread("./balck_img.jpg")
    arrow_img = cv2.imread("./ex6/ex63.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(img.shape)
    cv2.imwrite("black_img.jpg",img)
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    h,w,d = present_frame.shape
    #フレームの青い部分を二値化
    blue_threshold_before_img = image_processing.cut_blue_img1(before_frame)
    plt.imshow(blue_threshold_before_img)
    plt.show()
    blue_threshold_present_img = image_processing.cut_blue_img1(present_frame)
    plt.imshow(blue_threshold_present_img)
    plt.show()
    #コーナー検出
    try:
        before_p1,before_p2,before_p3,before_p4 = image_processing.points_extract1(blue_threshold_before_img,before_frame)
        present_p1,present_p2,present_p3,present_p4 = image_processing.points_extract1(blue_threshold_present_img,present_frame)
    except TypeError:
        print("Screen cannot be detected")
        return before_frame,[] ,[]
    #コーナーに従って画像の切り取り
    cut_present = present_frame[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    cut_before = before_frame[before_p1[1]:before_p2[1],before_p2[0]:before_p3[0]]
    #射影変換
    syaei_before_img = image_processing.projective_transformation(before_frame,before_p1,before_p2,before_p3,before_p4)
    plt.imshow(syaei_before_img)
    plt.show()
    syaei_present_img = image_processing.projective_transformation(present_frame,present_p1,present_p2,present_p3,present_p4)
    cv2.imwrite("present_frame.jpg",syaei_present_img)
    #対象画像をリサイズ
    syaei_before_img = cv2.resize(syaei_before_img,dsize=(610,211))
    syaei_present_img = cv2.resize(syaei_present_img,dsize=(610,211))
    #plt.imshow(syaei_present_img)
    #plt.show()
    #plt.imshow(syaei_before_img)
    #plt.show()
    #syaei_resize_before_img = cv2.resize(cut_before,dsize=(610,211))
    #syaei_resize_present_img = cv2.resize(syaei_present_img,dsize=(610,211))
    gray_present_img = cv2.cvtColor(syaei_present_img,cv2.COLOR_BGR2GRAY)
    gray_present_img = cv2.medianBlur(gray_present_img,3)
    ret, mask_present_img = cv2.threshold(gray_present_img,0,255,cv2.THRESH_OTSU)
    #膨張処理
    mask_present_img = cv2.dilate(mask_present_img,kernel)
    height_present,width_present = mask_present_img.shape
    array_present_H = image_processing.Projection_H(mask_present_img,height_present,width_present)
    presentH_THRESH = max(array_present_H)
    present_char_List = image_processing.Detect_HeightPosition(presentH_THRESH,height_present,array_present_H)
    present_char_List = np.reshape(present_char_List,[int(len(present_char_List)/2),2])
    #present_char_List = image_processing.convert_1d_to_2d(present_char_List,2)
    print(present_char_List)

    #plt.imshow(syaei_resize_present_img)
    #plt.show()
    #差分
    frame_diff = cv2.absdiff(blue_threshold_present_img,blue_threshold_before_img)
    #frame_diff = cv2.absdiff(present_frame,before_frame)
    #グレイスケール化
    gray_frame_diff = cv2.cvtColor(frame_diff,cv2.COLOR_BGR2GRAY)
    #ノイズ除去
    gray_frame_diff = cv2.medianBlur(gray_frame_diff,3)
    #二値画像へ
    ret, mask_frame_diff = cv2.threshold(gray_frame_diff,0,255,cv2.THRESH_OTSU)
    #膨張処理
    mask_frame_diff = cv2.dilate(mask_frame_diff,kernel)
    cv2.imwrite("frame_diff3.jpg",mask_frame_diff)
    #コーナーに従って画像の切り取り
    #cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]
    mask_cut_diff_frame = mask_frame_diff[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    cv2.imwrite("frame_diff4.jpg",mask_cut_diff_frame)
    height , width = mask_cut_diff_frame.shape
    array_H = image_processing.Projection_H(mask_cut_diff_frame,height,width)
    H_THRESH = max(array_H)
    char_List1 = image_processing.Detect_HeightPosition(H_THRESH,height,array_H)
    #char_List1 = image_processing.convert_1d_to_2d(char_List1,2)
    char_List1 = np.reshape(char_List1,[int(len(char_List1)/2),2])
    print(char_List1)
    #knn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(present_char_List) 
    #distances, indices = knn_model.kneighbors(char_List1)
    #print(indices)
    #indices = image_processing.get_unique_list(indices)
    #print(indices)
    engine = pyttsx3.init()
    before_frame_row = []
    sabun_count = 0
    output_text = []
    before_row1_arrow_exist = False
    before_row2_arrow_exist = False
    before_row3_arrow_exist = False
    before_row4_arrow_exist = False
    if arrow_exist(before_frame_row1):
        before_row1_arrow_exist = True
        height,width = before_frame_row1.shape
        #frame_row = cv2.medianBlur(frame_row,3)
        array_V = image_processing.Projection_V(before_frame_row1,height,width)
        W_THRESH = max(array_V)
        char_List = image_processing.Detect_WidthPosition(W_THRESH,width,array_V)
        before_arrow = before_frame_row1.copy()
        cv2.rectangle(before_arrow,(0,0),(int(char_List[1]+1),h-1),(0,0,0),-1)
        
    if arrow_exist(before_frame_row2):
        before_row2_arrow_exist = True
        height,width = before_frame_row2.shape
        #frame_row = cv2.medianBlur(frame_row,3)
        array_V = image_processing.Projection_V(before_frame_row2,height,width)
        W_THRESH = max(array_V)
        char_List = image_processing.Detect_WidthPosition(W_THRESH,width,array_V)
        before_arrow = before_frame_row2.copy()
        cv2.rectangle(before_arrow,(0,0),(int(char_List[1]+1),h-1),(0,0,0),-1)
    
    if arrow_exist(before_frame_row3):
        before_row3_arrow_exist = True
        height,width = before_frame_row1.shape
        #frame_row = cv2.medianBlur(frame_row,3)
        array_V = image_processing.Projection_V(before_frame_row3,height,width)
        W_THRESH = max(array_V)
        char_List = image_processing.Detect_WidthPosition(W_THRESH,width,array_V)
        before_arrow = before_frame_row3.copy()
        cv2.rectangle(before_arrow,(0,0),(int(char_List[1]+1),h-1),(0,0,0),-1)
    
    if arrow_exist(before_frame_row4):
        before_row4_arrow_exist = True
        height,width = before_frame_row1.shape
        #frame_row = cv2.medianBlur(frame_row,3)
        array_V = image_processing.Projection_V(before_frame_row4,height,width)
        W_THRESH = max(array_V)
        char_List = image_processing.Detect_WidthPosition(W_THRESH,width,array_V)
        before_arrow = before_frame_row4.copy()
        plt.imshow(before_arrow)
        plt.show()
        cv2.rectangle(before_arrow,(0,0),(int(char_List[1]+1),h-1),(0,0,0),-1)
    
    plt.imshow(before_arrow)
    plt.show()

    present_char_List , mask_present_img2 = image_processing.mask_make(blue_threshold_present_img)
    for i in present_char_List:
        normal = mask_present_img2.copy()
        cut_present = mask_present_img2[int(i[0]):int(i[1]),]
        #normal = cut_present.copy()
        cv2.rectangle(normal,(0,0),(w-1,int(i[0])-1),(0,0,0),-1)
        cv2.rectangle(normal,(0,int(i[1])-1),(w-1,h-1),(0,0,0),-1)
        plt.imshow(cut_present)
        plt.show()
        flag = arrow_exist(cut_present)
        print(flag)
        cut_present_img = syaei_present_img[int(i[0]):int(i[1]),]
        before_frame_row.append(cut_present)
        if not sabun(before_frame_row1,cut_present):
            #if (before_row1_arrow_exist == True) & (flag == False):
                #sabun_count = sabun_count -1
                #print("row1")
            sabun_count += 1

        if not sabun(before_frame_row2,cut_present):
            #if (before_row2_arrow_exist == True) & (flag == False):
                #sabun_count = sabun_count -1
                #print("row2")
            sabun_count += 1
    
        if not sabun(before_frame_row3,cut_present):
            #if (before_row3_arrow_exist == True) & (flag == False):
                #sabun_count = sabun_count -1
                #print("row3")
            sabun_count += 1
            
        if not sabun(before_frame_row4,cut_present):
            #if (before_row4_arrow_exist == True) & (flag == False):
                #sabun_count = sabun_count -1
                #print("row4")
            sabun_count += 1

        if sabun_count > 3:
            print(sabun_count)
            #plt.imshow(cut_present)
            #plt.show()
            if not sabun(before_arrow,cut_present):
                output_text_p,out = image_processing.match_text2(img_temp,label_temp,cut_present)
                output_text.append(out)
        
        sabun_count = 0
        #engine.runAndWait()
        #cv2,imwrite("yuu.jpg",cut_present_img)
    #if char_List1.size == 0: #差分がなければ
        #return False #音声出力しない
    #else:
        #present_img = present_frame
        #return True #音声出力する
    #sabun(img,cut_present_img)
    print(output_text)
    engine.say(output_text)
    if len(present_char_List) == 0:
        return output_text,img,img,img,img
    elif len(present_char_List) == 1:
        return output_text,before_frame_row[0] , img,img,img
    elif len(present_char_List) == 2:
        return output_text,before_frame_row[0] , before_frame_row[1] ,img,img
    elif len(present_char_List) == 3:
        return output_text,before_frame_row[0] , before_frame_row[1] ,before_frame_row[2] ,img
    elif len(present_char_List) == 4:
        return output_text,before_frame_row[0] , before_frame_row[1],before_frame_row[2],before_frame_row[3] 
        
def arrow_exist(frame_row):
    kernel = np.ones((3,3),np.uint8)
    arrow_img = cv2.imread("./arrow.jpg")
    arrow_img = cv2.cvtColor(arrow_img,cv2.COLOR_BGR2GRAY)
    arrow_img = cv2.resize(arrow_img,dsize=(26,36))

    height,width = frame_row.shape
    frame_row = cv2.medianBlur(frame_row,3)
    array_V = image_processing.Projection_V(frame_row,height,width)
    W_THRESH = max(array_V)
    char_List2 = image_processing.Detect_WidthPosition(W_THRESH,width,array_V)
    if len(char_List2) == 0:
        return False
    
    match_img = frame_row[:,int(char_List2[0])-1:int(char_List2[1])+1]
    
    try:
        match_img = cv2.resize(match_img,dsize=(26,36))
        match_img = cv2.dilate(match_img,kernel)
        #plt.imshow(match_img)
        #plt.show()
    except cv2.error:
        return False
    match = cv2.matchTemplate(match_img,arrow_img,cv2.TM_CCORR_NORMED)
    #返り値は最小類似点、最大類似点、最小の場所、最大の場所
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    print(max_value)
    if max_value > 0.8:#なぜかめっちゃ小さい
        return True
    else:
        return False

#列ごとに差分をとる
def sabun(before_frame_row,present_frame_row):
    kernel = np.ones((3,3),np.uint8)
    #gray_present_img = cv2.cvtColor(present_frame_row,cv2.COLOR_BGR2GRAY)
    present_frame_row = cv2.medianBlur(present_frame_row,3)
    #print(present_frame_row.shape)
    #ret, present_frame_row = cv2.threshold(gray_present_img,0,255,cv2.THRESH_OTSU)
    #膨張処理
    present_frame_row = cv2.dilate(present_frame_row,kernel)
    
    h ,w = present_frame_row.shape
    #print(before_frame_row.shape)
    before_frame_row = cv2.resize(before_frame_row,dsize=(w,h))
    before_frame_row = cv2.dilate(before_frame_row,kernel)
    #gray_before_img = cv2.cvtColor(before_frame_row,cv2.COLOR_BGR2GRAY)
    before_frame_row = cv2.medianBlur(before_frame_row,3)
    #ret, before_frame_row = cv2.threshold(gray_before_img,0,255,cv2.THRESH_OTSU)
    frame_diff = cv2.absdiff(present_frame_row,before_frame_row)
    frame_diff = cv2.medianBlur(frame_diff,5)
    #frame_diff = cv2.absdiff(present_frame,before_frame)
    #plt.imshow(frame_diff)
    #plt.show()
    height , width = frame_diff.shape
    array_V = image_processing.Projection_V(frame_diff,height,width)
    W_THRESH = max(array_V)
    char_List2 = image_processing.Detect_WidthPosition(W_THRESH,width,array_V)
    white_pixels1 = np.count_nonzero(present_frame_row)
    white_pixels2 = np.count_nonzero(before_frame_row)
    sum_white_pixels = white_pixels1 + white_pixels2
    white_pixels = np.count_nonzero(frame_diff)
    diff_white_pixels = sum_white_pixels - white_pixels
    if diff_white_pixels < 0:
        diff_white_pixels = - diff_white_pixels
        
    black_pixels = frame_diff.size - white_pixels
    print("前のフレームとの変化量%")
    #percent = white_pixels/frame_diff.size *100
    percent = white_pixels / sum_white_pixels * 100
    print(percent)
    if percent < 2:
        return True
    else:
        return False
def file_w(text,output_text):
    f = open('3dprint_window.txt',mode='a',encoding = 'UTF-8')
    f.write(text)
    f.write('\n')
    f.close()
        
def voice(output_text,voice_flag):
    #準備
    #文字認識
    #現在のカーソル
    present_kersol = audio_output.kersol_search(output_text)
    before = []
    after = []
    print(output_text)
    if len(present_kersol) == 0: # カーソルがない
        engine = pyttsx3.init()
        #rateはデフォルトが200
        voice = engine.getProperty('voices')
        engine.setProperty("voice",voice[1].id)
        rate = engine.getProperty('rate')
        engine.setProperty('rate',speed)
        #volume デフォルトは1.0 設定は0.0~1.0
        volume = engine.getProperty('volume')
        engine.setProperty('volume',vol)
        engine.say("window was changed to")
        audio_output.whole_text_read(output_text)
        engine.runAndWait()
        voice_flag.put(False) #音声終了
    
    else: #カーソルがあるとき
        audio_output.partial_text_read(present_kersol)
        engine.runAndWait()
        voice_flag.put(False)

    #前のテキストを保持
    #print(present_kersol)
    before_text = output_text
    before_kersol = present_kersol
    #file_w(out,output_text)

      

if __name__ == "__main__":
    #テンプレートをロード
    img1 = cv2.imread("./camera1/camera39.jpg")
    img2 = cv2.imread("./camera1/camera40.jpg")
    temp = np.load(r'./dataset2.npz')
    #テンプレート画像を格納
    img_temp = temp['x']
    #テンプレートのラベル(文)を格納
    label_temp = temp['y']
    before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4 =diff_image_search1(img1,img2,img_temp,label_temp)
    print("yy")
    plt.imshow(before_frame_row1)
    plt.show()
    voice_flag = multiprocessing.Queue()
    #voice_flagがTrueなら今発話中,Falseなら発話していない
    voice_flag.put(False)
    output_text , out= image_processing.match_text(img_temp,label_temp,img2)
    #voice1 = voice1 = multiprocessing.Process(target =voice,args = (output_text,voice_flag))
    #cv2.imwrite("before_frame_row1.jpg",before_frame_row1)
    #plt.imshow(before_frame_row2)
    #plt.show()
    #cv2.imwrite("before_frame_row2.jpg",before_frame_row2)
    #plt.imshow(before_frame_row3)
    #plt.show()
    #cv2.imwrite("before_frame_row3.jpg",before_frame_row3)
    #plt.imshow(before_frame_row4)
    #plt.show()
    #cv2.imwrite("before_frame_row4.jpg",before_frame_row4)

    
    output_text,before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4 =diff_image_search(img1,img2,img_temp,label_temp,before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4)
    #list(itertools.chain.from_iterable(output_text))
    #flat = [x for row in output_text for x in row]

    #voice1 = multiprocessing.Process(target =voice,args = (output_text,voice_flag))
    #voice1.start()