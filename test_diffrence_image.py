import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from MakeVoicefile.VoiceProcessing import text_read
from CharacterRecog.CharacterRecog import load_model, TextRecog
from ImageProcessing.img_processing2 import (
    sabun1,
    mask_make1,
    make_char_list,
    get_unique_list,
    mask_make,
    projective_transformation,
    points_extract1,
    cut_blue_img1,
    Projection_H,
    Detect_HeightPosition,
    cut_blue_img2,
)

before_frame = cv2.imread("./ProcessingDisplay/mask_frame_1699781826.7910953.jpg")

# frame = cv2.imread("./ProcessingDisplay/mask_frame_1699781826.9996154.jpg")
frame = cv2.imread("./ProcessingDisplay/mask_frame_1699781837.4042451.jpg")
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow("d", frame)
cv2.imshow("b", before_frame)
before_frame = cv2.cvtColor(before_frame, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)
model = cv2.bgsegm.createBackgroundSubtractorMOG()
<<<<<<< HEAD
=======
# for i in range(10):
>>>>>>> 1beaec42cd02ae959fd9e7d7615a35cf79e374d9
mask = model.apply(before_frame)
# print(last_insert_time)
# arrow_img = cv2.imread("./ex6/ex63.jpg")
# h,w,d = frame.shape
# フレームの青い部分を二値化
# frame= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
# cv2.imwrite("frameBE.jpg",before_frame)

# blue_threshold_present_img = cut_blue_img2(frame)
# cv2.imshow("blue_threshold_present_img", blue_threshold_present_img)
before_frame_row = []
sabun_count = 0
output_textx = ""
# present_char_List2, mask_present_img2 = mask_make(blue_threshold_present_img)
mask_frame = frame.copy()
before_frame = before_frame.astype("float")
l2 = 3

if l2 > 4:
    blue_threshold_present_img = cut_blue_img1(frame)
    mask_frame = mask_make1(blue_threshold_present_img)
    # blue = cut_blue_trans2(present_frame)
    mask = model.apply(mask_frame)
    # cv2.accumulateWeighted(mask_present_img2, before_frame, 0.8)
    # frame_diff = mask_present_img2 - cv2.convertScaleAbs(before_frame)
    # frame_diff[frame_diff ==205] = 0
    # frame_diff = cv2.absdiff(mask_present_img2,cv2.convertScaleAbs(before_frame))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    # frame_diff = cv2.dilate(frame_diff,kernel)
    # cv2.imwrite("./ProcessingDisplay/raaa.jpg",frame_diff)
    # cv2.imshow("mask", mask)
else:
    # blue = cut_blue_trans(present_frame)
    mask = model.apply(mask_frame)
    cv2.imshow("massssk", mask)
    # cv2.accumulateWeighted(mask_present_img2, before_frame, 0.8)
    # frame_diff = mask_present_img2 - cv2.convertScaleAbs(before_frame)
    # frame_diff[frame_diff == 205] = 0
    # frame_diff = cv2.absdiff(mask_present_img2,cv2.convertScaleAbs(before_frame))
    mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite("raaa.jpg",frame_diff)
# cv2.imshow("mask", mask)
mask_frame[mask == 0] = 0
cv2.imshow("mask_frame", mask_frame)
frame_diff = cv2.morphologyEx(mask_frame, cv2.MORPH_OPEN, kernel)

# plt.imshow(mask_present_img2)
# plt.show()
# h ,w = present_frame.shape
# print(before_frame_row.shape)
flg = 0
# before_frame = cv2.resize(before_frame,dsize=(w,h))
# contours, hierarchy = cv2.findContours(frame_diff.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# for i in range(len(contours)):
# if cv2.contourArea(contours[i]) < 10:
# frame_diff = cv2.fillPoly(frame_diff, [contours[i][:, 0, :]], (0, 255, 0), lineType=cv2.LINE_8, shift=0)
# cv2.imwrite("./ProcessingDisplay/realtimeimg_{0}.jpg".format(last_insert_time), frame_diff)
cv2.imshow("./ProcessingDisplay/mask_frame.jpg", frame)
# plt.imshow(frame_diff)
cv2.imshow("framediff.jpg", frame_diff)
cv2.waitKey(0)
