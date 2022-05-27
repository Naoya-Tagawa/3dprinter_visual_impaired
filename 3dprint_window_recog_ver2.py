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
import image_processing
import audio_output

event = threading.Event()
#flag = True: 音声出力
#flag = false: 音声出力しない

#話すスピード
speed = 150
#ボリューム
vol = 1.0

global present_frame
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
    output_text = []
    out = ""
    out_modify = ""
    s = {}
    new_d = {}
    #カーネル
    kernel = np.ones((3,3),np.uint8)
    #フレームの青い部分を二値化
    blue_threshold_before_img = image_processing.cut_blue_img(before_frame)
    blue_threshold_present_img = image_processing.cut_blue_img(present_frame)
    #コーナー検出
    try:
        before_p1,before_p2,before_p3,before_p4 = image_processing.points_extract(blue_threshold_before_img)
        present_p1,present_p2,present_p3,present_p4 = image_processing.points_extract(blue_threshold_present_img)
    except TypeError:
        print("Screen cannot be detected")
        return [] ,[]

    #コーナーに従って画像の切り取り
    #cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]]
    cut_present = present_frame[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    cut_before = before_frame[before_p1[1]:before_p2[1],before_p2[0]:before_p3[0]]
    #射影変換
    #syaei_before_img = syaei(before_frame,before_p1,before_p2,before_p3,before_p4)
    #syaei_present_img = syaei(present_frame,present_p1,present_p2,present_p3,present_p4)
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
    array_H = image_processing.Projection_H(mask_frame_diff,height,width)
    H_THRESH = max(array_H)
    char_List1 = image_processing.Detect_HeightPosition(H_THRESH,height,array_H)
    print(char_List1)
    for i in range(0,len(char_List1)-1,2):
        img_h = mask_frame_diff[int(char_List1[i]):int(char_List1[i+1]),:]
        img_j = cv2.rectangle(syaei_resize_present_img, (0 ,int(char_List1[i])), (610, int(char_List1[i+1])), (0,0,255), 2)
        height_h , width_h =img_h.shape
        #横方向のProjection Profileを得る
        array_V = image_processing.Projection_V(img_h,height_h,width_h)
        W_THRESH = max(array_V)
        char_List2 = image_processing.Detect_WidthPosition(W_THRESH,width_h,array_V)
        for j in range(0,len(char_List2)-1,2):
            #一文字ずつ切り取る
            #img_f = cv2.rectangle(syaei_resize_present_img, (int(char_List2[j]) ,int(char_List1[i])), (int(char_List2[j+1]), int(char_List1[i+1])), (0,0,255), 2)
            print("k")
    #cv2.imwrite("difference2.png",img_f)
    cv2.imwrite("diffecence3.jpg",img_j)
    if not char_List1: #差分がなければ
        event.clear() #音声出力しない
    else:
        present_img = present_frame
        event.set() #音声出力する

def voice():
    while True:
        if event.is_set(): #音声出力するかどうかチェック
            print("k")
            #output_text , out = image_processing.match_text(img_temp,label_temp,present_img)
            #現在のカーソル
            #present_kersol = audio_output.kersol_search(output_text)
        else:
            event.wait()
    before = []
    after = []
    #前と後のカーソルの類似度
    s = difflib.SequenceMatcher(None,before_kersol,present_kersol)
    #print(s.ratio())
    if audio_output.kersol_exist_search(before_kersol,out) == True: #前のカーソルがある(全画面変わっていない)
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
            audio_output.partial_text_read(before_kersol)
            engine.say("to")
            audio_output.partial_text_read(present_kersol)
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
            audio_output.whole_text_read(before)
            engine.say("to")
            audio_output.whole_text_read(after)
            engine.runAndWait()
        else:
            audio_output.whole_text_read(output_text)
            
    else: #全画面変化
        audio_output.whole_text_read(output_text)
        engine = pyttsx3.init()
        voice = engine.getProperty('voices')
        engine.setProperty("voice",voice[1].id)
        rate = engine.getProperty('rate')
        engine.setProperty('rate',speed)
        #volume デフォルトは1.0 設定は0.0~1.0
        volume = engine.getProperty('volume')
        engine.setProperty('volume',vol)
        if audio_output.kersol_exist_search == True:
            engine.say("The current cursor position is")
            audio_output.partial_text_read(present_kersol)
            engine.runAndWait()

    #前のテキストを保持
    print(present_kersol)
    before_text = output_text
    before_kersol = present_kersol
    #file_w(out,output_text)

      

if __name__ == "__main__":
    #対象画像をロード
    img1 = cv2.imread("./camera1/camera62.jpg")
    img2 = cv2.imread("./camera1/camera63.jpg")
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
    camera_thread = threading.Thread(target = camera)
    voice_thread = threading.Thread(target = voice)
    camera_thread.start()
    voice_thread.start()
    #match_text(img,before_text,kersol)
    #camera()
    diff_image_search(img1,img1)