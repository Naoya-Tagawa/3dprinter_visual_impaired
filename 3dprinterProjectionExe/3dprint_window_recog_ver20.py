
import multiprocessing
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
from ImageProcessing.img_processing2 import sabun1,cut_blue_trans2,mask_make1,make_char_list,get_unique_list,recog_text,projective_transformation2,cut_blue_trans,arrow_exist,mask_make, match_text3,projective_transformation,points_extract1,points_extract2,cut_blue_img1,Projection_H,Projection_V,Detect_HeightPosition,Detect_WidthPosition,match_text,match_text2,sabun,match,cut_blue_img2
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

#flag = True: 音声出力
#flag = false: 音声出力しない
CHUNK = 1024
#話すスピード
speed = 300
#ボリューム
vol = 10.0

def diff_image_search_first(present_frame,img_temp,label_temp,text_img):
    global present_img
    img = cv2.imread("./black_img.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w,d = present_frame.shape
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    #output_text = []
    out = ""
    #フレームの青い部分を二値化
    blue_threshold_present_img = cut_blue_img1(present_frame)
    
    #コーナー検出
    try:
        present_p1,present_p2,present_p3,present_p4 = points_extract1(blue_threshold_present_img)
    except TypeError:
        print("Screen cannot be detected")
        return img,img,img,img
    

    
    #コーナーに従って画像の切り取り
    cut_present = present_frame[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    
    #射影変換
    syaei_present_img = projective_transformation(present_frame,present_p1,present_p2,present_p3,present_p4)

    gray_present_img = cv2.cvtColor(syaei_present_img,cv2.COLOR_BGR2GRAY)
    gray_present_img = cv2.medianBlur(gray_present_img,3)
    ret, mask_present_img = cv2.threshold(gray_present_img,0,255,cv2.THRESH_OTSU)
    #膨張処理
    mask_present_img = cv2.dilate(mask_present_img,kernel)
    height_present,width_present = mask_present_img.shape
    array_present_H = Projection_H(mask_present_img,height_present,width_present)
    presentH_THRESH = max(array_present_H)
    present_char_List1 = Detect_HeightPosition(presentH_THRESH,height_present,array_present_H)
    present_char_List1 = np.reshape(present_char_List1,[int(len(present_char_List1)/2),2])
    


    #文字のみのマスク画像生成
    present_char_List2 , mask_present_img2 = mask_make(blue_threshold_present_img)
    before_frame_row = []
    #列ごとにマスク画像を取得
    for i in present_char_List2:
        normal = mask_present_img2.copy()
        cut_present_row = mask_present_img2[int(i[0]):int(i[1]),]
        before_frame_row.append(cut_present_row)
    
    
    if len(present_char_List2) == 0:
        return img,img,img,img,mask_present_img2
    elif len(present_char_List2) == 1:
        return before_frame_row[0] , img,img,img,mask_present_img2
    elif len(present_char_List2) == 2:
        return before_frame_row[0] , before_frame_row[1] ,img,img,mask_present_img2
    elif len(present_char_List2) == 3:
        return before_frame_row[0] , before_frame_row[1] ,before_frame_row[2] ,img,mask_present_img2
    elif len(present_char_List2) == 4:
        return before_frame_row[0] , before_frame_row[1],before_frame_row[2],before_frame_row[3],mask_present_img2
    else:
        return img,img,img,img,mask_present_img2


def diff_image_search(present_frame,before_frame,before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4,output_text,img_temp,label_temp):
    img = cv2.imread("./balck_img.jpg")
    kernel = np.ones((3,3),np.uint8)
    
    #arrow_img = cv2.imread("./ex6/ex63.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w,d = present_frame.shape
    #フレームの青い部分を二値化
    frame= cv2.cvtColor(present_frame,cv2.COLOR_BGR2RGB)
    cv2.imwrite("frameBE.jpg",before_frame)
    
    blue_threshold_present_img = cut_blue_img2(present_frame)
    #kk
    before_frame_row = []
    sabun_count = 0
    judge = False
    output_textx = ""
    count = 0
    present_char_List2,mask_present_img2 = mask_make(blue_threshold_present_img)
    
    before_frame = before_frame.astype('float')
        

    if len(present_char_List2) > 4:
        blue_threshold_present_img = cut_blue_img1(present_frame)
        mask_present_img2 = mask_make1(blue_threshold_present_img)
        #blue = cut_blue_trans2(present_frame)
        cv2.accumulateWeighted(mask_present_img2, before_frame, 0.8)
        frame_diff = mask_present_img2 - cv2.convertScaleAbs(before_frame)
        
        frame_diff[frame_diff ==205] = 0
        #frame_diff = cv2.absdiff(mask_present_img2,cv2.convertScaleAbs(before_frame))
        frame_diff = cv2.morphologyEx(frame_diff, cv2.MORPH_OPEN, kernel)
        frame_diff = cv2.medianBlur(frame_diff,3)
        #frame_diff = cv2.dilate(frame_diff,kernel)
        cv2.imwrite("raaa.jpg",frame_diff)
    else:
        #blue = cut_blue_trans(present_frame)
        cv2.accumulateWeighted(mask_present_img2, before_frame, 0.8)
        frame_diff = mask_present_img2 - cv2.convertScaleAbs(before_frame)
        frame_diff[frame_diff == 205] = 0
        #frame_diff = cv2.absdiff(mask_present_img2,cv2.convertScaleAbs(before_frame))
        frame_diff = cv2.medianBlur(frame_diff,3)
        frame_diff = cv2.morphologyEx(frame_diff, cv2.MORPH_OPEN, kernel)
        #cv2.imwrite("raaa.jpg",frame_diff)
#
#    cv2.imwrite("realtimeimg.jpg",frame_diff)

    #plt.imshow(mask_present_img2)
    #plt.show()
    #h ,w = present_frame.shape
    #print(before_frame_row.shape)
    flg = 0
    #before_frame = cv2.resize(before_frame,dsize=(w,h))
    contours, hierarchy = cv2.findContours(frame_diff.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #for i in range(len(contours)):
    #    if (cv2.contourArea(contours[i]) < 10):
    #        frame_diff = cv2.fillPoly(frame_diff, [contours[i][:,0,:]], (0,255,0), lineType=cv2.LINE_8, shift=0)
    #plt.imshow(frame_diff)
    cv2.imshow("framediff.jpg",frame_diff)
    cv2.imshow("before.jpg",before_frame)
    cv2.imshow("mas",mask_present_img2)
    #plt.show()
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    present_char_List1 = make_char_list(frame_diff)
    #try:
    
       # cv2.imshow("gg",blue)
        #cv2.waitKey(0)
        #p1,p2,p3,p4 = points_extract2(blue)
    #except TypeError:
        #print("Screen cannot be detected")

    #syaei_img,M = projective_transformation2(mask_present_img2,p1,p2,p3,p4)

    #print(present_char_List1)
    #List = [ [0,y] for l in present_char_List1 for y in l]
    #present_char_List2 = make_char_list(mask_present_img2)
    #print(present_char_List2)
    #List = [ [[0,y] for y in l ]for l in present_char_List1]
    #print(List)
    #pt = cv2.perspectiveTransform(np.array([List]),M)
    if len(present_char_List1) != 0:
        try:
            knn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(present_char_List2) 
            distances, indices = knn_model.kneighbors(present_char_List1)
            indices = get_unique_list(indices)
        except ValueError:
            indices = []
    else:
        indices = []
    #print(indices)
    for i in indices:
        
        if len(indices)==0:
            break
        elif len(indices) > 4:
            break
        cut_present = mask_present_img2[int(present_char_List2[i[0]][0]):int(present_char_List2[i[0]][1]),]
        if len(present_char_List2) == len(present_char_List1):
            before_frame_row.append(cut_present)
            flg = 1
        #if arrow_exist(cut_present):
            #cut_present,judge = arrow_exist_judge(cut_present)
        #cv2.imshow("HHH",cut_present)
        #cv2.waitKey(0)
        #before_frame_row.append(cut_present)
        if not sabun1(before_frame_row1,cut_present):
            sabun_count += 1

        if not sabun1(before_frame_row2,cut_present):
            sabun_count += 1
    
        if not sabun1(before_frame_row3,cut_present):
            sabun_count += 1
            
        if not sabun1(before_frame_row4,cut_present):
            sabun_count += 1
        
        #cut_present1 = mask_present_img[int(j[0]):int(j[1]),]
    
        if sabun_count > 3:
            out = match_text3(img_temp,label_temp,cut_present)
            #out = recog_text(cut_present)
            output_textx = output_textx + " \n" + out
            #before_frame_row.append(cut_present)
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
        #before_frame_row.append(cut_present)
        sabun_count = 0
        

        #count += 1
           
    if flg != 1:
        for i in present_char_List2:
            if len(present_char_List2)==0:
                break
            elif len(present_char_List2) > 4:
                break
            cut_present = mask_present_img2[int(i[0]):int(i[1]),]
            before_frame_row.append(cut_present)
    if len(output_textx)!=0:
        output_text.put(output_textx)

    #start1 = time.perf_counter()
    #end1 = time.perf_counter()
    #mask_present_img2,judge = arrow_exist_judge(mask_present_img2)
    try:
        if len(present_char_List2) == 0:
            return before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4,mask_present_img2
        elif len(present_char_List2) == 1:
            return before_frame_row[0] , before_frame_row2,before_frame_row3,before_frame_row4,mask_present_img2
        elif len(present_char_List2) == 2:
            return before_frame_row[0] , before_frame_row[1] ,before_frame_row3,before_frame_row4,mask_present_img2
        elif len(present_char_List2) == 3:
            return before_frame_row[0] , before_frame_row[1] ,before_frame_row[2] ,before_frame_row4,mask_present_img2
        elif len(present_char_List2) == 4:
            return before_frame_row[0] , before_frame_row[1],before_frame_row[2],before_frame_row[3],mask_present_img2
        else:
            return img,img,img,img,mask_present_img2
    except IndexError:
        return img,img,img,img,mask_present_img2

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

    #最初のフレームを取得する
    ret , bg = cap.read()
    before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4,before_frame= diff_image_search_first(bg,img_temp,label_temp,output_text)
    frame = bg
    count = 0
    
    h,w=frame.shape[:2]
    base=np.zeros((h,w,3),np.uint32)
    count = 0
    img_list = []
    for i in range(29):
        ret , frame = cap.read()
        base = base + frame
        img_list.append(frame)
        #make_img_file(frame)
    print(len(img_list))
    #before_frame = None
    while True:
        ret , frame = cap.read()
        #フレームが取得できない場合は画面を閉じる
        if not ret:
            cv2.destroyAllWindows()
        cv2.imshow("frame",frame)
        #画面が遷移したか調査
        #dst = cv2.bitwise_and(frame,frame,mask=before_frame)
        #for img in img_list:
            #base = base + img
        base = base + frame
        #base = base + frame + img_list[0]+ img_list[1] + img_list[2] + img_list[3] + img_list[4] + img_list[5] + img_list[6] + img_list[7] + img_list[8]+img_list[9]+img_list[10] + img_list[11] + img_list[12] + img_list[13]
        base1 = base/30
        #base1=np.where(base1<0,0,base1)
        #cv2.imwrite("base10.jpg",base)
        base1=base1.astype(np.uint8)
        cv2.imshow("base",base1)
        #cv2.imshow("baselast",img_list[0])
        #cv2.waitKey(0)
        before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4,before_frame= diff_image_search(base1,before_frame,before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4,output_text,img_temp,label_temp)
        #count  = 0
        #pre_img = get_pre_img()
        #cv2.imshow("ff",before_frame)
        #cv2.waitKey(0)
        base = base - img_list[0]
        img_list.pop(0)
        img_list.append(frame)
        #base=np.zeros((h,w,3),np.uint32)
        

        
        #diff_flag = Trueなら画面遷移,diff_flag=Falseなら画面遷移していない
        #present_kersol = audio_output.kersol_search(output_text)
        #if present_kersol == 1: # カーソルがない
        #qキーが入力されたら画面を閉じる
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        
    read.join()
    
    
    
    
    