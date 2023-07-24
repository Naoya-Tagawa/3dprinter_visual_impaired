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
from img_processing2 import make_char_list,get_unique_list,recog_text,projective_transformation2,cut_blue_trans,arrow_exist,mask_make, match_text3,projective_transformation,points_extract1,points_extract2,cut_blue_img1,Projection_H,Projection_V,Detect_HeightPosition,Detect_WidthPosition,match_text,match_text2,sabun,match,cut_blue_img2
import audio_output
from sklearn.neighbors import NearestNeighbors 
from io import BytesIO
def niBlackThreshold(src, blockSize, k):
    mean = cv2.boxFilter(src, cv2.CV_32F, (blockSize, blockSize), None, (-1, -1), True, cv2.BORDER_REPLICATE)
    sqmean = cv2.sqrBoxFilter(src, cv2.CV_32F, (blockSize, blockSize), None, (-1,-1), True, cv2.BORDER_REPLICATE)
    variance = sqmean - mean * mean
    stddev = cv2.sqrt(variance)
    thresh = mean + k * stddev
    diff = src - thresh
    diff[diff > 0.0] = 255.0
    diff[diff < 0.0] = 0.0
    dst = diff.astype(np.uint8)
    return dst

def nickThreshold(src, blockSize, k):
    mean = cv2.boxFilter(src, cv2.CV_32F, (blockSize,blockSize), None, (-1, -1), True, cv2.BORDER_REPLICATE)
    sqmean = cv2.sqrBoxFilter(src, cv2.CV_32F, (blockSize, blockSize), None, (-1,-1), True, cv2.BORDER_REPLICATE);
    variance = sqmean - mean * mean;
    sqrtVarianceMeanSum = cv2.sqrt(variance + sqmean);
    thresh = mean + k * sqrtVarianceMeanSum;
    diff = src - thresh
    diff[diff > 0.0] = 255.0
    diff[diff < 0.0] = 0.0
    dst = diff.astype(np.uint8)
    return dst
#テンプレートをロード
temp = np.load(r'./dataset2.npz')
#テンプレート画像を格納
img_temp = temp['x']
#テンプレートのラベル(文)を格納
label_temp = temp['y']
kernel = np.ones((3,3),np.uint8)
img4 = cv2.imread("./hei/camera518.jpg")
img = cv2.imread("./camera1/camera10.jpg")
plt.imshow(img4)
plt.show()
c = img4[223:252,]
h,w,d = img4.shape
hh= np.array([[223,252],[290,320]],dtype='float32')
List = [ [0,y] for l in hh for y in l]
print(List)
print(np.array(List))
print("ff")
hh= np.array([[0,223],[0,252],[0,290],[0,320]],dtype='float32')
hh = np.array([hh])
print(hh)
cv2.imshow("hhh",c)
cv2.waitKey(0)
blue_threshold_present_img = cut_blue_img2(img4)
present_char_List1 , mask_present_img2 = mask_make(blue_threshold_present_img)
#hh = np.array([[0,present_char_List1[0][0]],[0,present_char_List1[0][1]]],dtype='float32')
#hh = np.array([hh])

print(present_char_List1)
<<<<<<< HEAD
#for i in present_char_List1:
#    if len(present_char_List1)==0:
#            break
#    elif len(present_char_List1) > 4:
#            break
#    cut_present = mask_present_img2[int(i[0]):int(i[1]),]
#    cv2.imshow("p",cut_present)
#    cv2.waitKey(0)
#    out = match_text3(img_temp,label_temp,cut_present)
#    print(out)
cv2.imshow("syaei",mask_present_img2)
=======
for i in present_char_List1:
    if len(present_char_List1)==0:
            break
    elif len(present_char_List1) > 4:
            break
    cut_present = mask_present_img2[int(i[0]):int(i[1]),]
    cv2.imshow("p",cut_present)
    cv2.waitKey(0)
    out = match_text3(img_temp,label_temp,cut_present)
    print(out)
cv2.imshow("syaeil",mask_present_img2)
>>>>>>> b6d627b7d1231e7e1c302d1be9aae4fe4f881159
cv2.waitKey(0)
    #フレームの青い部分を二値化
blue_threshold_img = cut_blue_trans(img4)
b = cut_blue_img1(img)
<<<<<<< HEAD

# グレースケール変換
gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
#th, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
dst = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,101,20)
cv2.imshow("syaei99",dst)
cv2.waitKey(0)
cv2.imshow("syaei6",b)
=======
gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
  
# 方法2       
dst = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
cv2.imshow("dst",dst)
cv2.waitKey(0)
outa = recog_text(dst)
print(outa)
cv2.imshow("syaei",blue_threshold_img)
>>>>>>> b6d627b7d1231e7e1c302d1be9aae4fe4f881159
cv2.waitKey(0)
    #コーナー検出
try:
    p1,p2,p3,p4 = points_extract2(b)
except TypeError:
    print("Screen cannot be detected")
#p8,p5,p6,p7 = points_extract1(b)
print(p1,p2,p3,p4)
cv2.circle(img, p1, 3, (0,255,0), thickness=1, lineType=cv2.LINE_8, shift=0)
cv2.circle(img, p2, 3, (0,255,0), thickness=1, lineType=cv2.LINE_8, shift=0)
cv2.circle(img, p3, 3, (0,255,0), thickness=1, lineType=cv2.LINE_8, shift=0)
cv2.circle(img, p4, 3, (0,255,0), thickness=1, lineType=cv2.LINE_8, shift=0)
    #コーナーに従って画像の切り取り
#cut_img = window_img[p1[1]:p2[1],p2[0]:p3[0]]
cv2.imshow("syaei",img)
cv2.waitKey(0)
#射影変換
syaei_img,M = projective_transformation2(b,p1,p2,p3,p4)
pt = cv2.perspectiveTransform(hh,M)
print(pt)

syae = syaei_img[int(pt[0][0][1]):int(pt[0][1][1]),]
#syaei_img2 = projective_transformation(img,p8,p5,p6,p7)

<<<<<<< HEAD
gray = cv2.cvtColor(syaei_img, cv2.COLOR_RGB2GRAY)
dst = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
oute = recog_text(dst)
print(oute)
cv2.imshow("s",dst)
=======
cv2.imshow("syaeigg",syaei_img)
>>>>>>> b6d627b7d1231e7e1c302d1be9aae4fe4f881159
cv2.waitKey(0)
cv2.imshow("syaei",syae)
cv2.waitKey(0)
#out = match_text3(img_temp,label_temp,syae)
#ou = recog_text(syae)
#print(ou)
#print(out)


bin_niblack = nickThreshold(gray, 51, -0.1)

cv2.imshow("syaei",bin_niblack)
cv2.waitKey(0)
present_char_List1  = make_char_list(syaei_img)
print(present_char_List1)
#present_char_List1 = np.reshape(present_char_List1,[int(len(present_char_List1)/2),2])
#print(present_char_List1)
#cv2.imshow("syaei",syaei_img2)
#cv2.waitKey(0)
#pt = np.reshape(pt,[int(len(pt)/2),2])
pt = pt[0][:,1]
pt = np.reshape(pt,[int(len(pt)/2),2])
#pt = np.array([[pt[0][0][1],pt[0][1][1]],[pt[0][2][1],pt[0][3][1]]],dtype='float32')
print(pt)
#present_char_List1 = np.reshape(present_char_List1,[int(len(present_char_List1)/2),2])
knn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(present_char_List1) 
distances, indices = knn_model.kneighbors(pt)
indices = get_unique_list(indices)
print(indices)
for i in indices:
    cut_present_img = syaei_img[int(present_char_List1[i[0]][0]):int(present_char_List1[i[0]][1]),]
    cv2.imshow("cut",cut_present_img)
    cv2.waitKey(0)
#対象画像をリサイズ
out = match_text3(img_temp,label_temp,cut_present_img)
ou = recog_text(cut_present_img)
print(ou)
print(out)
img5 = cv2.imread("./kkwa.png")
blue_threshold_present_img = cut_blue_img1(img5)
present_char_List1 , mask_present_img2 = mask_make(blue_threshold_present_img)
cv2.imshow("ggh",mask_present_img2)
cv2.waitKey(0)
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
