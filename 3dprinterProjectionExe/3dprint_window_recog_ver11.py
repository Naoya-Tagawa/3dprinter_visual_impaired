from asyncio import futures
from msilib.schema import Error
import multiprocessing
from sys import _enablelegacywindowsfsencoding
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import glob
from natsort import natsorted
import multiprocessing
from PIL import Image , ImageTk , ImageOps
from pandas import cut
import pyttsx3 
from dictionary_word import speling
import numpy as np
import cv2
import ImageProcessing.img_processing2 as img_processing2
import audio_output
from threading import Thread
import threading
import time
import pyttsx3
from numpy import char
import multiprocessing
import os
event = multiprocessing.Event()
count = 0
import os
from operator import itemgetter
import datetime
import pyaudio
import wave
#flag = True: 音声出力
#flag = false: 音声出力しない
CHUNK = 1024
#話すスピード
speed = 300
#ボリューム
vol = 10.0

global present_img
global before_frame
def camera():
    global before_frame
    cap = cv2.VideoCapture(0)
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
    blue_threshold_present_img = img_processing2.cut_blue_img1(present_frame)
    
    #コーナー検出
    try:
        present_p1,present_p2,present_p3,present_p4 = img_processing2.points_extract1(blue_threshold_present_img,present_frame)
    except TypeError:
        print("Screen cannot be detected")
        return img,img,img,img
    
    #コーナーに従って画像の切り取り
    cut_present = present_frame[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    
    #射影変換
    syaei_present_img = img_processing2.projective_transformation(present_frame,present_p1,present_p2,present_p3,present_p4)

    gray_present_img = cv2.cvtColor(syaei_present_img,cv2.COLOR_BGR2GRAY)
    gray_present_img = cv2.medianBlur(gray_present_img,3)
    ret, mask_present_img = cv2.threshold(gray_present_img,0,255,cv2.THRESH_OTSU)
    #膨張処理
    mask_present_img = cv2.dilate(mask_present_img,kernel)
    height_present,width_present = mask_present_img.shape
    array_present_H = img_processing2.Projection_H(mask_present_img,height_present,width_present)
    presentH_THRESH = max(array_present_H)
    present_char_List1 = img_processing2.Detect_HeightPosition(presentH_THRESH,height_present,array_present_H)
    present_char_List1 = np.reshape(present_char_List1,[int(len(present_char_List1)/2),2])
    


    #文字のみのマスク画像生成
    present_char_List2 , mask_present_img2 = img_processing2.mask_make(blue_threshold_present_img)
    before_frame_row = []
    #列ごとにマスク画像を取得
    for i in present_char_List2:
        normal = mask_present_img2.copy()
        cut_present_row = mask_present_img2[int(i[0]):int(i[1]),]
        cv2.rectangle(normal,(0,0),(w-1,int(i[0])-1),(0,0,0),-1)
        cv2.rectangle(normal,(0,int(i[1])-1),(w-1,h-1),(0,0,0),-1)
        before_frame_row.append(cut_present_row)
    
    output_text_p, out = img_processing2.match_text(img_temp,label_temp,cut_present)
    output_text.put(output_text_p)
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
    #arrow_img = cv2.imread("./ex6/ex63.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    h,w,d = present_frame.shape
    #フレームの青い部分を二値化
    blue_threshold_present_img = img_processing2.cut_blue_img1(present_frame)
 
    before_frame_row = []
    sabun_count = 0

    if arrow_exist(before_frame_row1):
        before_arrow_exist = 1
        height,width = before_frame_row1.shape
        array_V = img_processing2.Projection_V(before_frame_row1,height,width)
        W_THRESH = max(array_V)
        char_List = img_processing2.Detect_WidthPosition(W_THRESH,width,array_V)
        before_arrow = before_frame_row1.copy()
        cv2.rectangle(before_arrow,(0,0),(int(char_List[1]+1),h-1),(0,0,0),-1)
        before_frame_row1 = before_arrow

    if arrow_exist(before_frame_row2):
        before_arrow_exist = 2
        height,width = before_frame_row2.shape
        array_V = img_processing2.Projection_V(before_frame_row2,height,width)
        W_THRESH = max(array_V)
        char_List = img_processing2.Detect_WidthPosition(W_THRESH,width,array_V)
        before_arrow = before_frame_row2.copy()
        cv2.rectangle(before_arrow,(0,0),(int(char_List[1]+1),h-1),(0,0,0),-1)
        before_frame_row2 = before_arrow
    
    if arrow_exist(before_frame_row3):
        before_arrow_exist = 3
        height,width = before_frame_row3.shape
        array_V = img_processing2.Projection_V(before_frame_row3,height,width)
        W_THRESH = max(array_V)
        char_List = img_processing2.Detect_WidthPosition(W_THRESH,width,array_V)
        before_arrow = before_frame_row3.copy()
        cv2.rectangle(before_arrow,(0,0),(int(char_List[1]+1),h-1),(0,0,0),-1)
        before_frame_row3 = before_arrow


    if arrow_exist(before_frame_row4):
        before_arrow_exist = 4
        height,width = before_frame_row4.shape
        array_V = img_processing2.Projection_V(before_frame_row4,height,width)
        W_THRESH = max(array_V)
        char_List = img_processing2.Detect_WidthPosition(W_THRESH,width,array_V)
        before_arrow = before_frame_row4.copy()
        cv2.rectangle(before_arrow,(0,0),(int(char_List[1]+1),h-1),(0,0,0),-1)
        before_frame_row4 = before_arrow
    
    count = 0
    output_textx = []
    present_char_List1 , mask_present_img2 = img_processing2.mask_make(blue_threshold_present_img)
    start = time.perf_counter()
    for i in present_char_List1:
        cut_present = mask_present_img2[int(i[0]):int(i[1]),]
        flag = arrow_exist(cut_present)
        if flag == True:
            height,width = cut_present.shape
            #frame_row = cv2.medianBlur(frame_row,3)
            array_V = img_processing2.Projection_V(cut_present,height,width)
            W_THRESH = max(array_V)
            char_List = img_processing2.Detect_WidthPosition(W_THRESH,width,array_V)
            cut_present_arrow = cut_present.copy()
            #plt.imshow(before_arrow)
            #plt.show()
            cv2.rectangle(cut_present_arrow,(0,0),(int(char_List[1]+1),h-1),(0,0,0),-1)
            cut_present1 = cut_present
            cut_present = cut_present_arrow

        #before_frame_row.append(cut_present)
        if not sabun(before_frame_row1,cut_present):
            sabun_count += 1

        if not sabun(before_frame_row2,cut_present):
            sabun_count += 1
    
        if not sabun(before_frame_row3,cut_present):
            sabun_count += 1
            
        if not sabun(before_frame_row4,cut_present):
            sabun_count += 1
        
        #cut_present1 = mask_present_img[int(j[0]):int(j[1]),]
    
        if sabun_count > 3:
            if flag == True:
                out = img_processing2.match_text3(img_temp,label_temp,cut_present1)
                output_textx.append(out)
                before_frame_row.append(cut_present1)
            else:
                out = img_processing2.match_text3(img_temp,label_temp,cut_present)
                output_textx.append(out)
                before_frame_row.append(cut_present)
            #try:
                #if not sabun(before_arrow,cut_present):
                    #output_text_p,out = img_processing2.match_text2(img_temp,label_temp,cut_present1)
                    #if out != "":
                        #output_textx.append(out)
            #except UnboundLocalError:
                    #output_text_p,out = img_processing2.match_text2(img_temp,label_temp,cut_present1)
                    #if out != "":
                        #output_textx.append(out)
        #矢印があるかどうか判定
        #if arrow_exist(cut_present):
        
    
        sabun_count = 0
        count += 1
    if len(output_textx) != 0:
        output_text.put(output_textx)
        voice_flag.value = 1
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
    h ,w = present_frame_row.shape
    before_frame_row = cv2.resize(before_frame_row,dsize=(w,h))
    frame_diff = cv2.absdiff(present_frame_row,before_frame_row)
    frame_diff = cv2.medianBlur(frame_diff,5)
    white_pixels1 = np.count_nonzero(present_frame_row)
    white_pixels2 = np.count_nonzero(before_frame_row)
    sum_white_pixels = white_pixels1 + white_pixels2
    white_pixels = np.count_nonzero(frame_diff)
    diff_white_pixels = sum_white_pixels - white_pixels
    if diff_white_pixels < 0:
        diff_white_pixels = - diff_white_pixels
        
    try:
        percent = white_pixels / sum_white_pixels * 100
    except ZeroDivisionError:
        percent = 100
    
    if percent < 2:
        print(percent)
        return True
    else:
        return False

def arrow_exist(frame_row):
    kernel = np.ones((3,3),np.uint8)
    arrow_img = cv2.imread("./arrow.jpg")
    arrow_img = cv2.cvtColor(arrow_img,cv2.COLOR_BGR2GRAY)
    arrow_img = cv2.resize(arrow_img,dsize=(26,36))

    height,width = frame_row.shape
    array_V = img_processing2.Projection_V(frame_row,height,width)
    W_THRESH = max(array_V)
    char_List2 = img_processing2.Detect_WidthPosition(W_THRESH,width,array_V)
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
    


def make_voice_file(text): #音声ファイル作成
    engine = pyttsx3.init()
    path = "./voice/"
    now = str(datetime.datetime.now())
    now_day , now_time = now.split()
    dh,m,s = now.split(':')
    sec , msec = s.split('.')
    now_time = sec + msec
    file_name = path + "voice_" + now_time + ".wav"
    print(file_name)
    engine.save_to_file(text,file_name)
    engine.runAndWait()

def delete_voice_file(): #音声ファイルを5つになるまで削除
    file_list = []
    path = "./voice/"
    for file in os.listdir("./voice"):
        base , ext = os.path.splitext(file)
        if ext == '.wav':
            wav_file = path + file
            file_list.append([file,os.path.getctime(wav_file)])
    file_list.sort(key = itemgetter(1),reverse=True)
    print(file_list)
    max_file = 3
    for i , file in enumerate(file_list):
        if i > max_file -1:
            wav_file = path + file[0]
            os.remove(wav_file)

def latest_play_voice_file(): #最新の音声ファイルを返す
    file_list = []
    path = "./voice/"
    for file in os.listdir("./voice"):
        base , ext = os.path.splitext(file)
        if ext == '.wav':
            wav_file = path + file
            file_list.append([file,os.path.getctime(wav_file)])
    file_list.sort(key = itemgetter(1),reverse=True)
    return path + file_list[0][0]

class AudioPlayer(object): #音声ファイルを再生、停止する
    """ A Class For Playing Audio """

    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.playing = threading.Event()    # 再生中フラグ

    def run(self):
        """ Play audio in a sub-thread """
        audio = pyaudio.PyAudio()
        input = wave.open(self.audio_file, "rb")
        output = audio.open(format=audio.get_format_from_width(input.getsampwidth()),
                            channels=input.getnchannels(),
                            rate=input.getframerate(),
                            output=True)

        while self.playing.is_set():
            data = input.readframes(CHUNK)
            if len(data) > 0:
                # play audio
                output.write(data)
            else:
                # end playing audio
                self.playing.clear()

        # stop and close the output stream
        output.stop_stream()
        output.close()
        # close the input file
        input.close()
        # close the PyAudio
        audio.terminate()

    def play(self):
        """ Play audio. """
        if not self.playing.is_set():
            self.playing.set()
            self.thread = threading.Thread(target=self.run)
            self.thread.start()

    def wait(self):
        if self.playing.is_set():
            self.thread.join()

    def stop(self):
        """ Stop playing audio and wait until the sub-thread terminates. """
        if self.playing.is_set():
            self.playing.clear()
            self.thread.join()


def text_read(output_text,voice_flag):
    start = 0
    while True:
        text = output_text.get()
        voice_flag.value = 0
        print(text)
        make_voice_file(text)
        file_name = latest_play_voice_file()
        if start !=0:
            player.stop()
        player = AudioPlayer(file_name)
        player.play()
        if start == 5:
            delete_voice_file()
            start = 1
        start += 1


if __name__ == "__main__":
    #テンプレートをロード
    temp = np.load(r'./dataset2.npz')
    #テンプレート画像を格納
    img_temp = temp['x']
    #テンプレートのラベル(文)を格納
    label_temp = temp['y']
    #diff_image_search(img1,img2)
    cap = cv2.VideoCapture(0)
    read_fps = cap.get(cv2.CAP_PROP_FPS)
    print(read_fps)
    voice_flag = multiprocessing.Value('i',0)
    #voice_flagが1なら今発話中,0なら発話していない
    count = 0
    output_text = multiprocessing.Queue()
    read = multiprocessing.Process(target=text_read,args=(output_text,voice_flag,))
    read.start()
    global t
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
        #qキーが入力されたら画面を閉じる
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        #time.sleep(0.1)