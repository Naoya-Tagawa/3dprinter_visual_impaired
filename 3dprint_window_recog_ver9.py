from asyncio import futures
from msilib.schema import Error
import multiprocessing
from sys import _enablelegacywindowsfsencoding
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
import image_processing
import audio_output
from concurrent.futures.process import ProcessPoolExecutor
import concurrent
from threading import Thread
#flag = True: 音声出力
#flag = false: 音声出力しない

#話すスピード
speed = 300
#ボリューム
vol = 10.0

global present_img
global before_frame 
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty("voice",voices[1].id)
#rateはデフォルトが200
rate = engine.getProperty('rate')
engine.setProperty('rate',speed)
#volume デフォルトは1.0 設定は0.0~1.0
volume = engine.getProperty('volume')
engine.setProperty('volume',vol)
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
        
def diff_image_search_first(present_frame,img_temp,label_temp,output_text):
    global present_img
    img = cv2.imread("./black_img.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w,d = present_frame.shape
    black_window = np.zeros((h,w))
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    #output_text = []
    out = ""
    #フレームの青い部分を二値化
    blue_threshold_present_img = image_processing.cut_blue_img1(present_frame)
    #cv2.imwrite("blue_threshold_present.jpg",blue_threshold_present_img)
    ##plt.imshow(blue_threshold_present_img)
    ##plt.show()
    #コーナー検出
    try:
        present_p1,present_p2,present_p3,present_p4 = image_processing.points_extract1(blue_threshold_present_img,present_frame)
    except TypeError:
        print("Screen cannot be detected")
        return img,img,img,img
    #before_frame = cv2.resize(before_frame,dsize=(610,211))
    #present_frame = cv2.resize(present_frame,dsize=(610,211))
    
    #コーナーに従って画像の切り取り
    cut_present = present_frame[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    
    #射影変換
    syaei_present_img = image_processing.projective_transformation(present_frame,present_p1,present_p2,present_p3,present_p4)
    #plt.imshow(syaei_present_img)
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
    present_char_List1 = image_processing.Detect_HeightPosition(presentH_THRESH,height_present,array_present_H)
    present_char_List1 = np.reshape(present_char_List1,[int(len(present_char_List1)/2),2])
    


    #文字のみのマスク画像生成
    present_char_List2 , mask_present_img2 = image_processing.mask_make(blue_threshold_present_img)
    #plt.imshow(mask_present_img2)
    #plt.show()
    #engine = pyttsx3.init()
    before_frame_row = []
    #列ごとにマスク画像を取得
    for i in present_char_List2:
        normal = mask_present_img2.copy()
        cut_present_row = mask_present_img2[int(i[0]):int(i[1]),]
        cv2.rectangle(normal,(0,0),(w-1,int(i[0])-1),(0,0,0),-1)
        cv2.rectangle(normal,(0,int(i[1])-1),(w-1,h-1),(0,0,0),-1)
        #plt.imshow(normal)
        #plt.show()
        before_frame_row.append(cut_present_row)
    #print(present_char_List2)
    #print("yes")
    output_text_p, out = image_processing.match_text(img_temp,label_temp,cut_present)
    output_text.put(output_text_p)
    #print(len(present_char_List2))
    if len(present_char_List2) == 0:
        return img,img,img,img
    elif len(present_char_List2) == 1:
        return before_frame_row[0] , img,img,img
    elif len(present_char_List2) == 2:
        return before_frame_row[0] , before_frame_row[1] ,img,img
    elif len(present_char_List2) == 3:
        return before_frame_row[0] , before_frame_row[1] ,before_frame_row[2] ,img
    elif len(present_char_List2) == 4:
        return before_frame_row[0] , before_frame_row[1],before_frame_row[2],before_frame_row[3] 
    else:
        return img,img,img,img


def diff_image_search(present_frame,img_temp,label_temp,before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4,output_text,voice_flag):
    global present_img
    img = cv2.imread("./balck_img.jpg")
    global term
    #arrow_img = cv2.imread("./ex6/ex63.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(img.shape)
    #cv2.imwrite("black_img.jpg",img)
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    h,w,d = present_frame.shape
    #フレームの青い部分を二値化
    #plt.imshow(blue_threshold_before_img)
    #plt.show()
    blue_threshold_present_img = image_processing.cut_blue_img1(present_frame)
    #plt.imshow(blue_threshold_present_img)
    #plt.show()
    #コーナー検出
    try:
        #before_p1,before_p2,before_p3,before_p4 = image_processing.points_extract1(blue_threshold_before_img,before_frame)
        present_p1,present_p2,present_p3,present_p4 = image_processing.points_extract1(blue_threshold_present_img,present_frame)
    except TypeError:
        print("Screen cannot be detected")
        return img,img,img,img
    #コーナーに従って画像の切り取り
    cut_present = present_frame[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    #cut_before = before_frame[before_p1[1]:before_p2[1],before_p2[0]:before_p3[0]]
    #射影変換
    #syaei_before_img = image_processing.projective_transformation(before_frame,before_p1,before_p2,before_p3,before_p4)
    #plt.imshow(syaei_before_img)
    #plt.show()
    syaei_present_img = image_processing.projective_transformation(present_frame,present_p1,present_p2,present_p3,present_p4)
    #対象画像をリサイズ
    #syaei_before_img = cv2.resize(syaei_before_img,dsize=(610,211))
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
    #print(present_char_List)

    ##plt.imshow(syaei_resize_present_img)
    ##plt.show()
    #差分
    #frame_diff = cv2.absdiff(blue_threshold_present_img,blue_threshold_before_img)
    #frame_diff = cv2.absdiff(present_frame,before_frame)
    #グレイスケール化
    #gray_frame_diff = cv2.cvtColor(frame_diff,cv2.COLOR_BGR2GRAY)
    #ノイズ除去
    #gray_frame_diff = cv2.medianBlur(gray_frame_diff,3)
    #二値画像へ
    #ret, mask_frame_diff = cv2.threshold(gray_frame_diff,0,255,cv2.THRESH_OTSU)
    #膨張処理
    #mask_frame_diff = cv2.dilate(mask_frame_diff,kernel)
    #cv2.imwrite("frame_diff3.jpg",mask_frame_diff)
    #コーナーに従って画像の切り取り
    #cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]
    #mask_cut_diff_frame = mask_frame_diff[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    #cv2.imwrite("frame_diff4.jpg",mask_cut_diff_frame)
    #height , width = mask_cut_diff_frame.shape
    #array_H = image_processing.Projection_H(mask_cut_diff_frame,height,width)
    #H_THRESH = max(array_H)
    #char_List1 = image_processing.Detect_HeightPosition(H_THRESH,height,array_H)
    #char_List1 = image_processing.convert_1d_to_2d(char_List1,2)
    #char_List1 = np.reshape(char_List1,[int(len(char_List1)/2),2])
    #print(char_List1)
    #knn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(present_char_List) 
    #distances, indices = knn_model.kneighbors(char_List1)
    #print(indices)
    #indices = image_processing.get_unique_list(indices)
    #print(indices)
    #engine = pyttsx3.init()
    before_frame_row = []
    sabun_count = 0
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
        height,width = before_frame_row3.shape
        #frame_row = cv2.medianBlur(frame_row,3)
        array_V = image_processing.Projection_V(before_frame_row3,height,width)
        W_THRESH = max(array_V)
        char_List = image_processing.Detect_WidthPosition(W_THRESH,width,array_V)
        before_arrow = before_frame_row3.copy()
        cv2.rectangle(before_arrow,(0,0),(int(char_List[1]+1),h-1),(0,0,0),-1)
    
    if arrow_exist(before_frame_row4):
        before_row4_arrow_exist = True
        height,width = before_frame_row4.shape
        #frame_row = cv2.medianBlur(frame_row,3)
        array_V = image_processing.Projection_V(before_frame_row4,height,width)
        W_THRESH = max(array_V)
        char_List = image_processing.Detect_WidthPosition(W_THRESH,width,array_V)
        before_arrow = before_frame_row4.copy()
        cv2.rectangle(before_arrow,(0,0),(int(char_List[1]+1),h-1),(0,0,0),-1)

    l = len(present_char_List)
    count = 0
    output_textx = []
    present_char_List1 , mask_present_img2 = image_processing.mask_make(blue_threshold_present_img)
    #print(present_char_List1)
    #print(present_char_List1)
    start = time.perf_counter()
    for (i,j) in zip(present_char_List1,present_char_List):
        if l == count:
            break 
        #normal = mask_present_img2.copy()
        cut_present = mask_present_img2[int(i[0]):int(i[1]),]
        #cv2.rectangle(normal,(0,0),(w-1,int(i[0])-1),(0,0,0),-1)
        #cv2.rectangle(normal,(0,int(i[1])-1),(w-1,h-1),(0,0,0),-1)
        #plt.imshow(cut_present)
        #plt.show()
        #flag = arrow_exist(cut_present)
        #cut_present_img = syaei_present_img[int(i[0]):int(i[1]),]
        before_frame_row.append(cut_present)
        if not sabun(before_frame_row1,cut_present):
            #if (before_row1_arrow_exist == True) & (flag == True):
                #sabun_count -= 1
            sabun_count += 1

        if not sabun(before_frame_row2,cut_present):
            #if (before_row2_arrow_exist == True) & (flag == True):
                #sabun_count -= 1
            sabun_count += 1
    
        if not sabun(before_frame_row3,cut_present):
            #if (before_row3_arrow_exist == True) & (flag == True):
                #sabun_count -= 1
            sabun_count += 1
            
        if not sabun(before_frame_row4,cut_present):
            #if (before_row4_arrow_exist == True) & (flag == True):
                #sabun_count -= 1
            sabun_count += 1
        #print("sabun_count = {0}".format(sabun_count))
        cut_present1 = mask_present_img[int(j[0]):int(j[1]),]
        #if (arrow_exist(cut_present1) == True) & (sabun_count < 4):
            #output_text_p,out = image_processing.match_text2(img_temp,label_temp,cut_present1)
            #output_text.append(out)
            #sabun_count = 0
            #count += 1
            #continue
        #end = time.perf_counter()
        #print("Rap time:{0}".format(end-start))
        if sabun_count > 3:
            #print(sabun_count)
            #plt.imshow(cut_present)
            #plt.show()
            #cv2.imwrite("cut_present.jpg",cut_present)
            #cut_present1 = mask_present_img[int(j[0]):int(j[1]),]
            try:
                if not sabun(before_arrow,cut_present):
                    output_text_p,out = image_processing.match_text2(img_temp,label_temp,cut_present1)
                    if out != "":
                        output_textx.append(out)
            except UnboundLocalError:
                    output_text_p,out = image_processing.match_text2(img_temp,label_temp,cut_present1)
                    if out != "":
                        output_textx.append(out)

        
    
        sabun_count = 0
        count += 1
    if len(output_textx) != 0:
        output_text.put(output_textx)
        voice_flag.value = 1
    #print(voice_flag.value)
        #engine.runAndWait()
        #cv2,imwrite("yuu.jpg",cut_present_img)
    #if char_List1.size == 0: #差分がなければ
        #return False #音声出力しない
    #else:
        #present_img = present_frame
        #return True #音声出力する
    #sabun(img,cut_present_img)
    #print(output_text)
    #engine.say(output_text)
    try:
        if len(present_char_List1) == 0:
            return img,img,img,img
        elif len(present_char_List1) == 1:
            return before_frame_row[0] , img,img,img
        elif len(present_char_List1) == 2:
            return before_frame_row[0] , before_frame_row[1] ,img,img
        elif len(present_char_List1) == 3:
            return before_frame_row[0] , before_frame_row[1] ,before_frame_row[2] ,img
        elif len(present_char_List1) == 4:
            return before_frame_row[0] , before_frame_row[1],before_frame_row[2],before_frame_row[3] 
        else:
            return img,img,img,img
    except IndexError:
        return img,img,img,img

#列ごとに差分をとる
def sabun(before_frame_row,present_frame_row):
    kernel = np.ones((3,3),np.uint8)
    #gray_present_img = cv2.cvtColor(present_frame_row,cv2.COLOR_BGR2GRAY)
    #present_frame_row = cv2.medianBlur(present_frame_row,3)
    #print(present_frame_row.shape)
    #ret, present_frame_row = cv2.threshold(gray_present_img,0,255,cv2.THRESH_OTSU)
    #膨張処理
    #present_frame_row = cv2.dilate(present_frame_row,kernel)
    
    h ,w = present_frame_row.shape
    #print(before_frame_row.shape)
    before_frame_row = cv2.resize(before_frame_row,dsize=(w,h))
    #before_frame_row = cv2.dilate(before_frame_row,kernel)
    #gray_before_img = cv2.cvtColor(before_frame_row,cv2.COLOR_BGR2GRAY)
    #before_frame_row = cv2.medianBlur(before_frame_row,3)
    #ret, before_frame_row = cv2.threshold(gray_before_img,0,255,cv2.THRESH_OTSU)
    frame_diff = cv2.absdiff(present_frame_row,before_frame_row)
    frame_diff = cv2.medianBlur(frame_diff,5)
    #frame_diff = cv2.absdiff(present_frame,before_frame)
    #plt.imshow(frame_diff)
    #plt.show()
    #cv2.imwrite("frame_diff.jpg",frame_diff)

    #height , width = frame_diff.shape
    #array_V = image_processing.Projection_V(frame_diff,height,width)
    #W_THRESH = max(array_V)
    #char_List2 = image_processing.Detect_WidthPosition(W_THRESH,width,array_V)
    white_pixels1 = np.count_nonzero(present_frame_row)
    white_pixels2 = np.count_nonzero(before_frame_row)
    sum_white_pixels = white_pixels1 + white_pixels2
    white_pixels = np.count_nonzero(frame_diff)
    diff_white_pixels = sum_white_pixels - white_pixels
    if diff_white_pixels < 0:
        diff_white_pixels = - diff_white_pixels
        
    #black_pixels = frame_diff.size - white_pixels
    ##percent = white_pixels/frame_diff.size *100
    try:
        percent = white_pixels / sum_white_pixels * 100
    except ZeroDivisionError:
        percent = 100
    #print(white_pixels)
    #print("%")
    if percent < 2:
        return True
    else:
        return False

def arrow_exist(frame_row):
    kernel = np.ones((3,3),np.uint8)
    arrow_img = cv2.imread("./arrow.jpg")
    arrow_img = cv2.cvtColor(arrow_img,cv2.COLOR_BGR2GRAY)
    arrow_img = cv2.resize(arrow_img,dsize=(26,36))
    #plt.imshow(arrow_img)
    #plt.show()
    height,width = frame_row.shape
    #frame_row = cv2.medianBlur(frame_row,3)
    array_V = image_processing.Projection_V(frame_row,height,width)
    W_THRESH = max(array_V)
    char_List2 = image_processing.Detect_WidthPosition(W_THRESH,width,array_V)
    if len(char_List2) == 0:
        return False
    
    match_img = frame_row[:,int(char_List2[0])-1:int(char_List2[1])+1]
    try:
        match_img = cv2.resize(match_img,dsize=(26,36))
        match_img = cv2.dilate(match_img,kernel)
    except cv2.error:
        return False
    match = cv2.matchTemplate(match_img,arrow_img,cv2.TM_CCORR_NORMED)
    #返り値は最小類似点、最大類似点、最小の場所、最大の場所
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    #print(max_value)
    if max_value > 0.6:#なぜかめっちゃ小さい
        return True
    else:
        return False
    
def voice(output_text):
    #準備
    #文字認識
    #output_text , out = image_processing.match_text(img_temp,label_temp,frame)
    #現在のカーソル
    #print("audio build")
    #present_kersol = audio_output.kersol_search(output_text)
    #if len(present_kersol) == 0: # カーソルがない
    audio_output.whole_text_read(output_text)
    #engine.runAndWait()
        #voice_flag.put(False) #音声終了
    
    #else: #カーソルがあるとき
        #audio_output.partial_text_read(present_kersol)
        #engine.runAndWait()
    #file_w(out,output_text)
    #print("voice finished")

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

def speak(phrase):
    engine = pyttsx3.init()
    engine.say(phrase)
    engine.runAndWait()
    engine.stop()

def stop_speaker():
    global term
    term = True
    t.join()

@threaded

def manage(p):
    global engine
    global term
    while p.is_alive():
        if term:
            engine.stop()
            term = False
        else:
            continue
def manage_process(p):
	global term
	while p.is_alive():
		if term:
			p.terminate()
			term = False
		else:
			continue
def speak_stop(voice_flag):
    if voice_flag == 1:
        print("stop")
        engine.stop()
        
def text_read(output_text,voice_flag):
    #global engine
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty("voice",voices[1].id)
    #rateはデフォルトが200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',speed)
    #volume デフォルトは1.0 設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',vol)
    while True:
        text = output_text.get()
        voice_flag.value = 0
        print(text)
        print("flag :{0}".format(voice_flag.value))
        print("queu size :{0}".format(output_text.qsize()))
        if output_text.qsize() >= 2:
            while output_text.qsize() > 1:
                text = output_text.get()
        for word in text:
            if output_text.qsize() >= 1:
                while output_text.qsize() > 0:
                    text = output_text.get()
                break
            #print(voice_flag.value)
            #if voice_flag.value == 1:
                #print("stop")
                #voice_flag.value = 0
                #break
            #engine.say(word)
            engine.connect('started-word', speak_stop(voice_flag))
            print(word)
            engine.say(word)

        engine.runAndWait()
        engine.stop()

def sy(phrase):
    global t 
    global term
    term = False
    global engine
    engine = pyttsx3.init()
    p = Thread(target=text_read,args=(phrase,engine))
    p.start()
    t = manage(p)
def say(phrase):
	global t
	global term
	term = False
	p = multiprocessing.Process(target=text_read, args=(phrase,))
	p.start()
	t = manage_process(p)

if __name__ == "__main__":
    #テンプレートをロード
    img1 = cv2.imread("./camera1/camera62.jpg")
    img2 = cv2.imread("./camera1/camera63.jpg")
    temp = np.load(r'./dataset2.npz')
    #テンプレート画像を格納
    img_temp = temp['x']
    #テンプレートのラベル(文)を格納
    label_temp = temp['y']
    #diff_image_search(img1,img2)
    cap = cv2.VideoCapture(1)
    read_fps = cap.get(cv2.CAP_PROP_FPS)
    print(read_fps)
    voice_flag = multiprocessing.Value('i',0)
    #voice_flagが1なら今発話中,0なら発話していない
    count = 0
    output_text = multiprocessing.Queue()
    read = multiprocessing.Process(target=text_read,args=(output_text,voice_flag,))
    read.start()
    global t
    global term
    #最初のフレームを取得する
    ret , bg = cap.read()
    before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4 = diff_image_search_first(bg,img_temp,label_temp,output_text)
    frame = bg
    
    while True:
        #start = time.perf_counter()
        ret , frame = cap.read()
        #フレームが取得できない場合は画面を閉じる
        if not ret:
            cv2.destroyAllWindows()
        cv2.imshow("frame",frame)
        #画面が遷移したか調査
        start = time.perf_counter()
        before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4= diff_image_search(frame,img_temp,label_temp,before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4,output_text,voice_flag)
        #diff_flag = Trueなら画面遷移,diff_flag=Falseなら画面遷移していない
        #present_kersol = audio_output.kersol_search(output_text)
        #if present_kersol == 1: # カーソルがない
        #end = time.perf_counter()
        #print("time :{0}".format(end-start))
        #qキーが入力されたら画面を閉じる
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        time.sleep(0.1)
        #time.sleep(1)