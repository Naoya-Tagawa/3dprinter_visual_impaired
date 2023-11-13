import cv2
import ImageProcessing.img_processing2 as img_processing2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import itertools
img = cv2.imread("./hei/camera186.jpg")
img1 = cv2.imread("./hei/camera181.jpg")
img2 = cv2.imread("./hei/camera120.jpg")
img3 = cv2.imread("./hei/camera618.jpg")
img4 = cv2.imread("./hei/camera518.jpg")
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
h,w,d = img1.shape
    #フレームの青い部分を二値化
img1 = cv2.resize(img1, img.shape[1::-1])
blue_threshold_present_img = img_processing2.cut_blue_img2(img4)
present_char_List1 , mask_present_img2 = img_processing2.mask_make(blue_threshold_present_img)
cv2.imshow("hh",mask_present_img2)
cv2.waitKey(0)
#mask_present_img2 = cv2.resize(mask_present_img2, img.shape[1::-1])
#mask_present_img2 = cv2.cvtColor(mask_present_img2, cv2.COLOR_GRAY2BGR)
blue_threshold_present_img1 = img_processing2.cut_blue_img2(img)
present_char_List1 , mask_present_img3 = img_processing2.mask_make(blue_threshold_present_img1)
print(mask_present_img2.shape)
cv2.imwrite("mask.jpg",mask_present_img3)
dst1 = cv2.bitwise_and(img3,img3,mask=mask_present_img2)
cv2.imshow("dst1",dst1)
cv2.waitKey(0)
#cv2.imwrite("mask_p.jpg",dst)

present_char_List1 , mask_present_img2 = img_processing2.mask_make(blue_threshold_present_img)
hist_mask = cv2.calcHist([img],[0],mask_present_img2,[256],[0,256])
color = ('b','g','r')

#dst = cv2.imread("./mask_p.jpg")

dst2 = cv2.bitwise_and(img2,img2,mask=mask_present_img2)
cv2.imshow("dst2",dst2)
cv2.waitKey(0)
cv2.imwrite("dst.jpg",dst2)
dst1[dst1 >= 255] = 0
dst2[dst2>= 255] = 0
h,w,e = dst1.shape
print(h*w)
#count1 =  sum(((r>0) and (g>0) and (b>0)) for d in dst1 for r,g,b in d)
#count2 =  sum(((r>0) and (g>0) and (b>0)) for d in dst2 for r,g,b in d)
dst0 = list(itertools.chain.from_iterable(dst1))
dst3 = list(itertools.chain.from_iterable(dst2))
count1 =  sum(((r>0) and (g>0) and (b>0)) for r,g,b in dst0)
count2 =  sum(((r>0) and (g>0) and (b>0)) for r,g,b in dst3)
dst1_count = sum(((b>0) and (r>150)) for b,g,r in dst0)
dst2_count = sum(((b>0) and (r>150)) for b,g,r in dst3)
#dst1_count = sum(((b>0) and (r>150)) for d in dst1 for b,g,r in d)
#dst2_count = sum(((b>0) and (r>150)) for d in dst2 for b,g,r in d)
print(count1)
print(count2)
print(dst1_count)
print(dst2_count)

for i,col in enumerate(color):
    histr = cv2.calcHist([dst1],[i],mask_present_img2,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    #histr += histr
plt.savefig("hist6.png")
plt.show()
for i,col in enumerate(color):
    histr = cv2.calcHist([dst2],[i],mask_present_img2,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])




plt.savefig("hist7.png")
plt.show()


