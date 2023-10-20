import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
from img_deal import sabun1,cut_blue_trans2,mask_make1,make_char_list,get_unique_list,recog_text,projective_transformation2,cut_blue_trans,arrow_exist,mask_make, match_text3,projective_transformation,points_extract1,points_extract2,cut_blue_img1,Projection_H,Projection_V,Detect_HeightPosition,Detect_WidthPosition,match_text,match_text2,sabun,match,cut_blue_img2
from PIL import Image

  
if __name__ == "__main__":
    count = 0
    files = glob.glob('./chara_add/*')
    for f in files:
        print(f)
        # input image
        img = cv2.imread(f)
        #cv2.imshow("img",img)
        #cv2.waitKey(0)
        #対象画像をロード
        #青い部分のみを二値化
        #cv2.imshow("close",img)
        #cv2.waitKey(0)
        close_img = cut_blue_img2(img)
        #cv2.imshow("close",close_img)
        #cv2.waitKey(0)
        close_img = mask_make1(close_img)
        #cv2.imshow("close",close_img)
        #cv2.waitKey(0)
        #コーナー検出
        p1,p2,p3,p4 = points_extract1(close_img)
        #コーナーに従って画像の切り取り
        #img_k = img[p1[1]:p2[1],p2[0]:p3[0]]
        #射影変換
        #syaei_img = projective_transformation(img,p1,p2,p3,p4)
        #cv2.imshow("syaei",syaei_img)
        #cv2.waitKey(0)
        # convert gray scale image
        #kernel = np.ones((3,3),np.uint8)
        #gray_img = cv2.cvtColor(syaei_img, cv2.COLOR_RGB2GRAY)
       # cv2.imshow("gray",gray_img)
        #cv2.waitKey(0)
        ## black white
        #ret, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
        #cv2.imshow("bw",bw_img)
        #cv2.waitKey(0)
        ##ノイズ除去
        #img_mask = cv2.medianBlur(bw_img,3)
        #膨張化
        #img_mask = cv2.dilate(bw_img,kernel)
        height, width = close_img.shape
        #cv2.imshow("img_mask",close_img)
        #cv2.waitKey(0)
        # create projection distribution
        array_H = Projection_H(close_img, height, width)
        #array_V = Projection_V(bw_img, height, width)
 
        # detect character height position
        H_THRESH = max(array_H)
        char_List1 = Detect_HeightPosition(H_THRESH, height, array_H)
        print(char_List1)
        # detect character width position
        #W_THRESH = max(array_V)
        #char_List2 = Detect_WidthPosition(W_THRESH, width, array_V)
        #print(array_V)
        # draw image
        if (len(char_List1) % 2) == 0:
            k=0
            #print("Succeeded in character detection")
            for i in range(0,len(char_List1)-1,2):
                img_h = close_img[int(char_List1[i]):int(char_List1[i+1]),:]
                h , w = img_h.shape
                array_V = Projection_V(img_h,h,w)
                W_THRESH = max(array_V)
                char_List2 = Detect_WidthPosition(W_THRESH,w,array_V)
                #print(char_List2)
                for j in range(0,len(char_List2)-1, 2):
                    img_f = close_img[int(char_List1[i])-1:int(char_List1[i+1])+1, int(char_List2[j])-1:int(char_List2[j+1])+1]
                    #cv2.imwrite("result{0}.jpg".format(k),img_f)
                    #k += 1
                    #cv2.imshow("img_f",img_f)
                    #cv2.waitKey(0)
                    cv2.imwrite("./chara/add_data/ex{0}.jpg".format(count), img_f)
                    count += 1
        
        else:
            print("Failed to detect characters")