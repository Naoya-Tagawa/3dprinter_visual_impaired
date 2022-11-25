import cv2
from img_processing2 import arrow_exist_judge,make_char_list,arrow_exist,mask_make, match_text3,projective_transformation,points_extract1,cut_blue_img1,Projection_H,Projection_V,Detect_HeightPosition,Detect_WidthPosition,match_text,match_text2,sabun,cut_blue_img2,recog_text,mask_make1
import numpy as np
cap = cv2.VideoCapture(0)
wait_secs = int(1000 / cap.get(cv2.CAP_PROP_FPS))

model = cv2.createBackgroundSubtractorMOG2()
before = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.imread("./balck_img.jpg")
    kernel = np.ones((3,3),np.uint8)
    #arrow_img = cv2.imread("./ex6/ex63.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w,d = frame.shape
    #フレームの青い部分を二値化
    blue_threshold_present_img = cut_blue_img2(frame)
    #kk
    before_frame_row = []
    sabun_count = 0
    judge = False
    output_textx = ""
    count = 0
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.accumulateWeighted(frame, before, 0.8)
    frame_diff = cv2.absdiff(frame,cv2.convertScaleAbs(before))
    if before is None:
        before = frame.copy().astype('float')
        continue
    
    present_char_List1 , mask_present_img2 = mask_make(blue_threshold_present_img)

    
    if len(present_char_List1) > 4:
        blue_threshold_present_img = cut_blue_img1(frame)
        mask_present_img2 = mask_make1(blue_threshold_present_img)
        #cv2.accumulateWeighted(mask_present_img2, before, 0.8)
        #frame_diff = cv2.absdiff(mask_present_img2,cv2.convertScaleAbs(before))
        #frame_diff = cv2.medianBlur(frame_diff,3)
        #frame_diff = cv2.dilate(frame_diff,kernel)
    
        #cv2.accumulateWeighted(mask_present_img2, before, 0.8)
        #frame_diff = cv2.absdiff(mask_present_img2,cv2.convertScaleAbs(before))
        #frame_diff = cv2.medianBlur(frame_diff,3)
        
        #mask = cv2.medianBlur(mask,3)
    # 背景の画素は黒 (0, 0, 0) にする。
    #mask_present_img2[mask == 0] = 0
    cv2.imshow("F", mask_present_img2)
    cv2.imshow("Frame (Only Forground)", frame_diff)
    cv2.waitKey(wait_secs)

cap.release()
cv2.destroyAllWindows()