from ImageProcessing.img_processing2 import arrow_exist,mask_make,projective_transformation,points_extract1,cut_blue_img1,Projection_H,Projection_V,Detect_HeightPosition,Detect_WidthPosition,match_text,match_text2,sabun
import cv2
import matplotlib
import numpy as np
kernel = np.ones((3,3),np.uint8)
img = cv2.imread("./camera1/camera45.jpg")
print("高さ")
print(img.shape[1])
blue_threshold_present_img = cut_blue_img1(img)
    
    #コーナー検出
present_p1,present_p2,present_p3,present_p4 = points_extract1(blue_threshold_present_img)
    #コーナーに従って画像の切り取り
cut_present = img[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]

syaei_present_img = projective_transformation(img,present_p1,present_p2,present_p3,present_p4)
syaei_present_img = cv2.resize(syaei_present_img,dsize=(610,211))
gray_present_img = cv2.cvtColor(syaei_present_img,cv2.COLOR_BGR2GRAY)
gray_present_img = cv2.medianBlur(gray_present_img,3)
ret, mask_present_img = cv2.threshold(gray_present_img,0,255,cv2.THRESH_OTSU)
mask_present_img = cv2.dilate(mask_present_img,kernel)
height_present,width_present = mask_present_img.shape
array_present_H = Projection_V(mask_present_img,height_present,width_present)
print(array_present_H)
#array_present_l = Projection_V2(mask_present_img,height_present,width_present)
#print(array_present_l)
presentH_THRESH = max(array_present_H)
present_char_List = Detect_HeightPosition(presentH_THRESH,height_present,array_present_H)
present_char_List = np.reshape(present_char_List,[int(len(present_char_List)/2),2])


cv2.imshow("kk",cut_present)
cv2.waitKey(0)
cv2.destroyAllWindows()
    #射影変換