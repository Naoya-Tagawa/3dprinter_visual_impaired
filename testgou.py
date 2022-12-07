import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
from img_processing2 import cut_blue_trans2,mask_make1,make_char_list,get_unique_list,recog_text,projective_transformation2,cut_blue_trans,arrow_exist,mask_make, match_text3,projective_transformation,points_extract1,points_extract2,cut_blue_img1,Projection_H,Projection_V,Detect_HeightPosition,Detect_WidthPosition,match_text,match_text2,sabun,match,cut_blue_img2
from natsort import natsorted
import multiprocessing
from pandas import cut
import pyttsx3 
import numpy as np
from threading import Thread
import threading
import time
import pyttsx3
from numpy import char
import multiprocessing
event = multiprocessing.Event()
count = 0
import os
from operator import itemgetter
import datetime
import pyaudio
import wave
from sklearn.neighbors import NearestNeighbors 
CHUNK = 1024
#話すスピード
speed = 300
#ボリューム
vol = 10.0


def make_voice_file(text): #音声ファイル作成
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty("voice", voices[1].id)
    path = "./voice/"
    now = str(datetime.datetime.now())
    now_day , now_time = now.split()
    dh,m,s = now.split(':')
    sec , msec = s.split('.')
    now_time = sec + msec
    file_name = path + "voice_" + now_time + ".wav"
    #print(file_name)
    engine.save_to_file(text,file_name)
    engine.runAndWait()


def make_img_file(img): #音声ファイル作成
    
    path = "./ave_img/"
    now = str(datetime.datetime.now())
    now_day , now_time = now.split()
    dh,m,s = now.split(':')
    sec , msec = s.split('.')
    now_time = sec + msec
    file_name = path + "ave_img_" + now_time + ".jpg"
    #print(file_name)
    cv2.imwrite(file_name,img)

def get_pre_img(): #最古の音声ファイルを返す
    file_list = []
    path = "./ave_img/"
    for file in os.listdir("./ave_img"):
        base , ext = os.path.splitext(file)
        if ext == '.jpg':
            wav_file = path + file
            file_list.append([file,os.path.getctime(wav_file)])
    file_list.sort(key = itemgetter(1),reverse=False)
    pre_img = cv2.imread(path + file_list[0][0])
    pre_img=pre_img.astype(np.float32)
    os.remove(path + file_list[0][0])
    return pre_img
def delete_all_file(): #音声ファイルを削除
    file_list = []
    path = "./ave_img/"
    for file in os.listdir("./ave_img"):
        base , ext = os.path.splitext(file)
        if ext == '.jpg':
            wav_file = path + file
            file_list.append([file,os.path.getctime(wav_file)])
    file_list.sort(key = itemgetter(1),reverse=True)
    #print(file_list)
    for i , file in enumerate(file_list):
        file_name = path + file[0]
        os.remove(file_name)

def delete_voice_file(): #音声ファイルを5つになるまで削除
    file_list = []
    path = "./voice/"
    for file in os.listdir("./voice"):
        base , ext = os.path.splitext(file)
        if ext == '.wav':
            wav_file = path + file
            file_list.append([file,os.path.getctime(wav_file)])
    file_list.sort(key = itemgetter(1),reverse=True)
    #print(file_list)
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


def text_read(output_text,img_temp,label_temp):
    start = 0
    while True:
        text = output_text.get()
        #cv2.imwrite("real.jpg",img)
        print("queu size :{0}".format(output_text.qsize()))
        if output_text.qsize() >= 1:
            while output_text.qsize() > 1:
                text = output_text.get()
                
                #cv2.imwrite("real.jpg",img)
        
        #out = match_text3(img_temp,label_temp,img)
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
        #time.sleep(0.1)


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
    #text_img = multiprocessing.Queue()
    output_text = multiprocessing.Queue()
    read = multiprocessing.Process(target=text_read,args=(output_text,img_temp,label_temp))
    read.start()
    delete_all_file()
    #最初のフレームを取得する
    ret , bg = cap.read()
    before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4,before_frame= diff_image_search_first(bg,img_temp,label_temp,output_text)
    frame = bg
    count = 0
    
    h,w=frame.shape[:2]
    base=np.zeros((h,w,3),np.uint32)
    for i in range(9):
        base = base + frame
        make_img_file(frame)
    
    #before_frame = None
    while True:
        ret , frame = cap.read()
        #フレームが取得できない場合は画面を閉じる
        if not ret:
            cv2.destroyAllWindows()
        cv2.imshow("frame",frame)
        #画面が遷移したか調査

        base = frame+ base
        base1 = base/10
        
        
        base1=base1.astype(np.uint8)
        cv2.imwrite("base.jpg",base1)
        before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4,before_frame= diff_image_search(base1,before_frame,before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4,output_text,img_temp,label_temp)
        #count  = 0
        pre_img = get_pre_img()
        base = base - pre_img
        cv2.imwrite("baseb.jpg",base/9)
        make_img_file(frame)
        
        

        
        #diff_flag = Trueなら画面遷移,diff_flag=Falseなら画面遷移していない
        #present_kersol = audio_output.kersol_search(output_text)
        #if present_kersol == 1: # カーソルがない
        #qキーが入力されたら画面を閉じる
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        
    read.join()