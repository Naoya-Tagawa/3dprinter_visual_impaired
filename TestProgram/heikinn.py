#ライブラリのインポート
import cv2
import numpy as np
import glob
from ImageProcessing.img_processing2 import arrow_exist_judge,make_char_list,arrow_exist,mask_make, match_text3,projective_transformation,points_extract1,cut_blue_img1,Projection_H,Projection_V,Detect_HeightPosition,Detect_WidthPosition,match_text,match_text2,sabun,cut_blue_img2,recog_text,mask_make1
#情報収集
pics='./hei/*.jpg'
print(len(pics))
pic_list=glob.glob(pics)
img=cv2.imread(pic_list[0],cv2.IMREAD_GRAYSCALE)
img = cv2.imread("./hei/camera104.jpg")
img1 = cv2.imread("./hei/camera105.jpg")
img2 = cv2.imread("./hei/camera106.jpg")
img3 = cv2.imread("./hei/camera107.jpg")
img4 = cv2.imread("./hei/camera108.jpg")
h,w=img.shape[:2]
base_array=np.zeros((h,w,3),np.uint32)
base_array = base_array+img+img1+img2+img3+img4
#平均画像の作成
count=1
for pic in pic_list:
    img0=cv2.imread(pic)
    blue_threshold_present_img = cut_blue_img2(img0)
    present_char_List1 , mask_present_img2 = mask_make(blue_threshold_present_img)
    #cv2.imshow("{0}".format(pic),mask_present_img2)
    count+=1
    #cv2.waitKey(0)
base_array=base_array/5

base_array=base_array.astype(np.uint8)
cv2.imwrite('avg.jpg',base_array)
blue_threshold_present_img = cut_blue_img2(img)
present_char_List1 , mask_present_img2 = mask_make(blue_threshold_present_img)
cv2.imshow("mm0",mask_present_img2)
cv2.waitKey(0)

blue_threshold_present_img = cut_blue_img2(base_array)
present_char_List1 , mask_present_img2 = mask_make(blue_threshold_present_img)
cv2.imshow("複合",mask_present_img2)
cv2.waitKey(0)