import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from natsort import natsorted
def read_image():
    x = [] #画像データ用
    y = [] #らべる
    #ファイルの保存先
    outfile = r'C:\Users\Naoya Tagawa\Desktop\dataset.npz'
    def glob_files(img_path,label_path,w,h):
        files = glob.glob(img_path)
        files = natsorted(files)
        num = 0
        for f in files:
            if num >= len(files):
                break
            num += 1
            img = Image.open(f)  #Pillow(PIL)で画像読み込み。
            img = np.asarray(img) #ndarray化
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #グレイスケール化
            ret, img = cv2.threshold(img,125,255,cv2.THRESH_BINARY) #二値化
            img = cv2.resize(img,dsize=(w,h))
            #画像データ、ラベルデータを保存
            x.append(img)
            fa = open(label_path+'\ex{0}.txt'.format(num),'r',encoding = 'UTF-8')
            data = fa.read()
            y.append(data)
            fa.close()

    glob_files(r'C:\Users\Naoya Tagawa\Desktop\ex2\*',r'C:\Users\Naoya Tagawa\Desktop\ex2tex',640,480)
    np.savez(outfile,x=x,y=y)



