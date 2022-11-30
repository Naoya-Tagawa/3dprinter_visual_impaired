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
from image_processing import cut_blue_img
from img_processing2 import recog_text,projective_transformation2,cut_blue_trans,arrow_exist,mask_make, match_text3,projective_transformation,points_extract1,points_extract2,cut_blue_img1,Projection_H,Projection_V,Detect_HeightPosition,Detect_WidthPosition,match_text,match_text2,sabun,match,cut_blue_img2
import audio_output
from sklearn.neighbors import NearestNeighbors 
from io import BytesIO

#テンプレートをロード
temp = np.load(r'./dataset2.npz')
#テンプレート画像を格納
img_temp = temp['x']
#テンプレートのラベル(文)を格納
label_temp = temp['y']
kernel = np.ones((3,3),np.uint8)
img4 = cv2.imread("./hei/camera1077.jpg")
img = cv2.imread("./camera1/camera10.jpg")
plt.imshow(img4)
plt.show()
c = img4[223:252,]
h,w,d = img4.shape
hh= np.array([[0,223],[0,252]],dtype='float32')
hh = np.array([hh])
cv2.imshow("hhh",c)
cv2.waitKey(0)
blue_threshold_present_img = cut_blue_img2(img4)
present_char_List1 , mask_present_img2 = mask_make(blue_threshold_present_img)
#hh = np.array([[0,present_char_List1[0][0]],[0,present_char_List1[0][1]]],dtype='float32')
#hh = np.array([hh])

print(present_char_List1)
for i in present_char_List1:
    if len(present_char_List1)==0:
            break
    elif len(present_char_List1) > 4:
            break
    cut_present = mask_present_img2[int(i[0]):int(i[1]),]
    cv2.imshow("p",cut_present)
    cv2.waitKey(0)
cv2.imshow("syaei",mask_present_img2)
cv2.waitKey(0)
    #フレームの青い部分を二値化
blue_threshold_img = cut_blue_trans(img4)
b = cut_blue_img1(img)
cv2.imshow("syaei",b)
cv2.waitKey(0)
cv2.imshow("syaei",blue_threshold_img)
cv2.waitKey(0)
    #コーナー検出
try:
    p1,p2,p3,p4 = points_extract2(blue_threshold_img)
except TypeError:
    print("Screen cannot be detected")
#p8,p5,p6,p7 = points_extract1(b)
print(p1,p2,p3,p4)
cv2.circle(img4, p1, 3, (0,255,0), thickness=1, lineType=cv2.LINE_8, shift=0)
cv2.circle(img4, p2, 3, (0,255,0), thickness=1, lineType=cv2.LINE_8, shift=0)
cv2.circle(img4, p3, 3, (0,255,0), thickness=1, lineType=cv2.LINE_8, shift=0)
cv2.circle(img4, p4, 3, (0,255,0), thickness=1, lineType=cv2.LINE_8, shift=0)
    #コーナーに従って画像の切り取り
#cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]]
cv2.imshow("syaei",img4)
cv2.waitKey(0)
#射影変換
syaei_img,M = projective_transformation2(mask_present_img2,p1,p2,p3,p4)
pt = cv2.perspectiveTransform(hh,M)
print(pt)
syae = syaei_img[int(pt[0][0][1]):int(pt[0][1][1]),]
#syaei_img2 = projective_transformation(img,p8,p5,p6,p7)

cv2.imshow("syaei",syaei_img)
cv2.waitKey(0)
cv2.imshow("syaei",syae)
cv2.waitKey(0)
out = match_text3(img_temp,label_temp,syae)
ou = recog_text(syae)
print(ou)
print(out)

cv2.imshow("syaei",syaei_img)
cv2.waitKey(0)
present_char_List1 , mask_present_img2 = mask_make(syaei_img)
#cv2.imshow("syaei",syaei_img2)
#cv2.waitKey(0)
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
