
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

#テンプレートをロード
temp = np.load(r'./dataset2.npz')
#テンプレート画像を格納
img_temp = temp['x']
#テンプレートのラベル(文)を格納
label_temp = temp['y']
kernel = np.ones((3,3),np.uint8)
before_frame = cv2.imread("./camera1/camera55.jpg")
img = cv2.imread("./camera1/camera56.jpg")
cv2.imshow("img",img)
cv2.waitKey(0)
blue_threshold_present_img = cut_blue_img1(img)
present_char_List2 , mask_present_img2 = mask_make(blue_threshold_present_img)
cv2.imshow("img",mask_present_img2)
cv2.waitKey(0)
blue_threshold_present_img1 = cut_blue_img1(before_frame)
present_char_List1 , before_frame = mask_make(blue_threshold_present_img1)
count =0
for i in present_char_List1:
    cut_present = mask_present_img2[int(i[0]):int(i[1]),]
    count +=1 
    cv2.imwrite("cut_mask{0}.jpg".format(count),cut_present)
cv2.imwrite("mask_power.jpg",mask_present_img2)
cv2.imwrite("mask_before.jpg",before_frame)
before_frame = before_frame.astype('float')
cv2.accumulateWeighted(mask_present_img2, before_frame, 0.5)
frame_diff = cv2.absdiff(mask_present_img2,cv2.convertScaleAbs(before_frame))
frame_diff = cv2.medianBlur(frame_diff,3)
cv2.imshow("img",frame_diff)
cv2.waitKey(0)
cv2.imwrite("mask_diff.jpg",frame_diff)
present_char_List1 = make_char_list(frame_diff)
knn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(present_char_List2) 
distances, indices = knn_model.kneighbors(present_char_List1)
indices = get_unique_list(indices)
for i in indices:
        if len(present_char_List1)==0:
            break
        elif len(present_char_List1) > 4:
            break
        cut_present = mask_present_img2[int(present_char_List2[i[0]][0]):int(present_char_List2[i[0]][1]),0:220]
        cv2.imwrite("cut_power.jpg",cut_present)
        cv2.imshow("img",cut_present)
        cv2.waitKey(0)
        #out = match_text3(img_temp,label_temp,cut_present)
        out = recog_text(cut_present)

print(out)