from img_processing2 import arrow_exist,mask_make,projective_transformation,points_extract1,cut_blue_img1,Projection_H,Projection_V,Detect_HeightPosition,Detect_WidthPosition,match_text,match_text2,sabun
import cv2
import matplotlib
img = cv2.imread("./camera1/camera45.jpg")
blue_threshold_present_img = cut_blue_img1(img)
    
    #コーナー検出
present_p1,present_p2,present_p3,present_p4 = points_extract1(blue_threshold_present_img,img)
    #コーナーに従って画像の切り取り
cut_present = img[present_p1[1]:present_p2[1],present_p2[0]:present_p3[0]]

cv2.imshow("kk",cut_present)
cv2.waitKey(0)
cv2.destroyAllWindows()
    #射影変換