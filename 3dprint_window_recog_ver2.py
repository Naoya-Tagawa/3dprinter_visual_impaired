import multiprocessing
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
#flag = True: 音声出力
#flag = false: 音声出力しない

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


def diff_image_search(before_frame,present_frame):
    global present_img
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    #フレームの青い部分を二値化
    #blue_threshold_before_img = image_processing.cut_blue_img(before_frame)
    blue_threshold_present_img = image_processing.cut_blue_img(present_frame)
    #コーナー検出
    try:
        #before_p1,before_p2,before_p3,before_p4 = image_processing.points_extract(blue_threshold_before_img)
        present_p1,present_p2,present_p3,present_p4 = image_processing.points_extract(blue_threshold_present_img)
    except TypeError:
        print("Screen cannot be detected")
        return False,[],[],[]

    #コーナーに従って画像の切り取り
    #cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]]
    cut_present = present_frame[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    #cut_before = before_frame[before_p1[1]:before_p2[1],before_p2[0]:before_p3[0]]
    #射影変換
    #syaei_before_img = syaei(before_frame,before_p1,before_p2,before_p3,before_p4)
    syaei_present_img = image_processing.projective_transformation(present_frame,present_p1,present_p2,present_p3,present_p4)
    #対象画像をリサイズ
    gray_present_img = cv2.cvtColor(syaei_present_img,cv2.COLOR_BGR2GRAY)
    gray_present_img = cv2.medianBlur(gray_present_img,3)
    ret, mask_present_img = cv2.threshold(gray_present_img,0,255,cv2.THRESH_OTSU)
    mask_present_img = cv2.dilate(mask_present_img,kernel)
    height_present,width_present = mask_present_img.shape
    array_present_H = image_processing.Projection_H(mask_present_img,height_present,width_present)
    presentH_THRESH = max(array_present_H)
    present_char_List = image_processing.Detect_HeightPosition(presentH_THRESH,height_present,array_present_H)
    print(present_char_List)
    present_char_List = np.reshape(present_char_List,[int(len(present_char_List)/2),2])
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
    char_List1 = np.reshape(char_List1,[int(len(char_List1)/2),2])
    print(char_List1)
    #knn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(present_char_List) 
    #distances, indices = knn_model.kneighbors(char_List1)
    #print(indices)
    #indices = image_processing.get_unique_list(indices)
    #print(indices)
    #for i in indices:
        #img_j = cv2.rectangle(syaei_present_img, (0 ,int(present_char_List[i[0]][0])), (610, int(present_char_List[i[0]][1])), (0,0,255), 2)
    #cv2.imwrite("diffecence3.jpg",img_j)
    
    if char_List1.size == 0: #差分がなければ
        return False,present_char_List,[],syaei_present_img #音声出力しない
    else:
        knn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(present_char_List) 
        distances, indices = knn_model.kneighbors(char_List1)
        indices = image_processing.get_unique_list(indices)

        return True,present_char_List,indices,syaei_present_img #音声出力する

def voice(frame,voice_flag,present_char_List,indices,img_temp,label_temp,present_img):
    start = time.perf_counter()
    engine = pyttsx3.init()
    #rateはデフォルトが200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',speed)
    #volume デフォルトは1.0 設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',vol)
    #文字認識
    output_text , out = image_processing.match_text(img_temp,label_temp,frame)
    cv2.imwrite("present.jpg",frame)
    #現在のカーソル
    end = time.perf_counter()
    print(end-start)
    present_kersol = audio_output.kersol_search(output_text)
    if len(present_kersol) == 0: # カーソルがない
        for i in indices:
            print("カーソルなし")
            cut_present_img = present_img[int(present_char_List[i[0]][0]):int(present_char_List[i[0]][1]),]
            cv2.imwrite("cut_present.jpg",cut_present_img)
            output_text,out = image_processing.match_text2(img_temp,label_temp,cut_present_img)
            #print(out)
            audio_output.partial_text_read(output_text)
            end = time.perf_counter()
            print(end-start)
            #engine.runAndWait()
            
        voice_flag.put(False) #音声終了

    
    else: #カーソルがあるとき
        audio_output.partial_text_read(present_kersol)
        voice_flag.put(False)

    #前のテキストを保持
    print(present_kersol)
    #file_w(out,output_text)

def first_voice(frame,voice_flag,img_temp,label_temp):
    output_text,out = image_processing.match_text(img_temp,label_temp,frame)
    audio_output.whole_text_read(output_text)
    voice_flag.put(False)


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
    cap = cv2.VideoCapture(0)
    read_fps = cap.get(cv2.CAP_PROP_FPS)
    print(read_fps)
    voice_flag = multiprocessing.Queue()
    voice_flag.put(False)
    #voice_flagがTrueなら今発話中,Falseなら発話していない
    count = 0
    while True:
        start = time.perf_counter()
        ret , frame = cap.read()
        #フレームが取得できない場合は画面を閉じる
        if not ret:
            cv2.destroyAllWindows()
        cv2.imshow("frame",frame)
        if count == 0:
            count += 1
            voice1 = multiprocessing.Process(target =first_voice,args = (frame,voice_flag,img_temp,label_temp))
            voice1.start()
            voice_flag.put(True)
            before_frame = frame
            end = time.perf_counter()
            print(end-start)
            continue

        #画面が遷移したか調査
        diff_flag,present_char_List,indices,present_img= diff_image_search(before_frame,frame)
        end = time.perf_counter()
        print(end-start)
        #diff_flag = Trueなら画面遷移,diff_flag=Falseなら画面遷移していない
        before_frame = frame
        if diff_flag == True:
            st = voice_flag.get()
            if st == True:
                print("pp")
                #voice1.terminate()
            #voice1 = multiprocessing.Process(target=voice,args=(frame,voice_flag,present_char_List,indices,img_temp,label_temp,present_img))
            #voice1.start()
            voice(frame,voice_flag,present_char_List,indices,img_temp,label_temp,present_img)
            voice_flag.put(True)
        end = time.perf_counter()
        print(end-start)
        #qキーが入力されたら画面を閉じる
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
        #time.sleep(1)