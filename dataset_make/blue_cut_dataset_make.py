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
import ImageProcessing.image_processing as image_processing
from sklearn.neighbors import NearestNeighbors 
def diff_image_search(present_frame,img_temp,label_temp,before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4):
    global present_img
    img = cv2.imread("./balck_img.jpg")
    arrow_img = cv2.imread("./ex6/ex63.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(img.shape)
    cv2.imwrite("black_img.jpg",img)
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
        return [] ,img,img,img,img
    #コーナーに従って画像の切り取り
    cut_present = present_frame[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]
    syaei_present_img = image_processing.projective_transformation(present_frame,present_p1,present_p2,present_p3,present_p4)
    syaei_present_img = cv2.resize(syaei_present_img,dsize=(610,211))
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
    print(present_char_List)

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
    engine = pyttsx3.init()
    before_frame_row = []
    sabun_count = 0
    output_text = []
    before_row1_arrow_exist = False
    before_row2_arrow_exist = False
    before_row3_arrow_exist = False
    before_row4_arrow_exist = False

    print(output_text)
    engine.say(output_text)
    if len(present_char_List) == 0:
        return output_text,img,img,img,img
    elif len(present_char_List) == 1:
        return output_text,before_frame_row[0] , img,img,img
    elif len(present_char_List) == 2:
        return output_text,before_frame_row[0] , before_frame_row[1] ,img,img
    elif len(present_char_List) == 3:
        return output_text,before_frame_row[0] , before_frame_row[1] ,before_frame_row[2] ,img
    elif len(present_char_List) == 4:
        return output_text,before_frame_row[0] , before_frame_row[1],before_frame_row[2],before_frame_row[3] 
    else:
        return output_text,img,img,img,img