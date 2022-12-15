import cv2
from img_processing2 import arrow_exist_judge,make_char_list,arrow_exist,mask_make, match_text3,projective_transformation,points_extract1,cut_blue_img1,Projection_H,Projection_V,Detect_HeightPosition,Detect_WidthPosition,match_text,match_text2,sabun,cut_blue_img2,recog_text,mask_make1
import numpy as np
cap = cv2.VideoCapture(0)
wait_secs = int(1000 / cap.get(cv2.CAP_PROP_FPS))

model = cv2.createBackgroundSubtractorMOG2(history=30,detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.imread("./balck_img.jpg")
    kernel = np.ones((3,3),np.uint8)
    #arrow_img = cv2.imread("./ex6/ex63.jpg")
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w,d = frame.shape
    #フレームの青い部分を二値化
    #blue_threshold_present_img = cut_blue_img2(frame)
    #kk
    #before_frame_row = []
    ##sabun_count = 0
    #judge = False
    #output_textx = ""
    #count = 0
    #present_char_List1 , mask_present_img2 = mask_make(blue_threshold_present_img)

    #if len(present_char_List1) > 4:
    #    blue_threshold_present_img = cut_blue_img1(frame)
    #    mask_present_img2 = mask_make1(blue_threshold_present_img)
    mask = model.apply(frame)
        #mask = cv2.medianBlur(mask,3)
    #    mask = cv2.dilate(mask,kernel)
    #else:
    #    mask_present_img2 = mask_make1(blue_threshold_present_img)
    #    mask = model.apply(mask_present_img2)
    #    mask = cv2.dilate(mask,kernel)
        #mask = cv2.medianBlur(mask,3)
    # 背景の画素は黒 (0, 0, 0) にする。
<<<<<<< HEAD
    frame[mask == 0] = 0
    #contours, hierarchy = cv2.findContours(frame.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #for i in range(len(contours)):
        #if (cv2.contourArea(contours[i]) < 30):
        #    frame = cv2.fillPoly(frame, [contours[i][:,0,:]], (0,255,0), lineType=cv2.LINE_8, shift=0)
=======
    mask_present_img2[mask == 0] = 0
<<<<<<< HEAD
>>>>>>> f34da87169d4f759c22b00ab7dc2022c7bcad797

    cv2.imshow("Frame (Only Forground)", frame)
=======
    mask_present_img2 = cv2.morphologyEx(mask_present_img2, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(mask_present_img2.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if (cv2.contourArea(contours[i]) < 30):
            mask_present_img2 = cv2.fillPoly(mask_present_img2, [contours[i][:,0,:]], (0,255,0), lineType=cv2.LINE_8, shift=0)
<<<<<<< HEAD
    cv2.imshow("Frame (Only Forground)", mask)
>>>>>>> 9ffbb6bf42f5505f4665b5faf5a5113fe412d8b4
=======
    cv2.imshow("Frame (Only Forground)", mask_present_img2)
>>>>>>> e15bea30be865aa63b93406a6dc4edb2b7a1e7e1
    cv2.waitKey(wait_secs)

cap.release()
cv2.destroyAllWindows()
