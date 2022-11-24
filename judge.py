import cv2
import img_processing2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img = cv2.imread("./hei/camera186.jpg")
img1 = cv2.imread("./hei/camera181.jpg")
img2 = cv2.imread("./hei/camera120.jpg")

#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
h,w,d = img1.shape
    #フレームの青い部分を二値化
img1 = cv2.resize(img1, img.shape[1::-1])
blue_threshold_present_img = img_processing2.cut_blue_img2(img1)
present_char_List1 , mask_present_img2 = img_processing2.mask_make(blue_threshold_present_img)
cv2.imshow("hh",mask_present_img2)
cv2.waitKey(0)
#mask_present_img2 = cv2.resize(mask_present_img2, img.shape[1::-1])
#mask_present_img2 = cv2.cvtColor(mask_present_img2, cv2.COLOR_GRAY2BGR)
blue_threshold_present_img1 = img_processing2.cut_blue_img2(img)
present_char_List1 , mask_present_img3 = img_processing2.mask_make(blue_threshold_present_img1)
print(mask_present_img2.shape)
cv2.imwrite("mask.jpg",mask_present_img3)
dst1 = cv2.bitwise_and(img,img,mask=mask_present_img2)
cv2.imshow("hh",dst1)
cv2.waitKey(0)
#cv2.imwrite("mask_p.jpg",dst)

present_char_List1 , mask_present_img2 = img_processing2.mask_make(blue_threshold_present_img)
hist_mask = cv2.calcHist([img],[0],mask_present_img2,[256],[0,256])
color = ('b','g','r')

dst = cv2.imread("./mask_p.jpg")

dst2 = cv2.bitwise_and(img2,img2,mask=mask_present_img2)
cv2.imshow("hh",dst2)
cv2.waitKey(0)
cv2.imwrite("dst.jpg",dst2)
#dst1[dst1 >= 255] = 0
#dst2[dst2>= 255] = 0

count1 =  sum(((r>0) and (g>0) and (b>0)) for d in dst1 for r,g,b in d)
count2 =  sum(((r>0) and (g>0) and (b>0)) for d in dst2 for r,g,b in d)
dst1_count = sum(((g>200) and (r>150)) for d in dst1 for r,g,b in d)
dst2_count = sum(((g>200) and (r>150)) for d in dst2 for r,g,b in d)
print(count1)
print(count2)
print(dst1_count)
print(dst2_count)

for i,col in enumerate(color):
    histr = cv2.calcHist([dst1],[i],None,[256],[0,256])
    
    #histr += histr

plt.plot(histr,color = 'g')
plt.xlim([0,256])



plt.savefig("hist6.png")
plt.show()


