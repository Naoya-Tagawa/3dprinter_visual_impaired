import cv2
import img_processing2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img = cv2.imread("./hei/camera186.jpg")
img1 = cv2.imread("./hei/camera181.jpg")
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
h,w,d = img1.shape
    #フレームの青い部分を二値化
img1 = cv2.resize(img1, img.shape[1::-1])
blue_threshold_present_img = img_processing2.cut_blue_img2(img)


present_char_List1 , mask_present_img2 = img_processing2.mask_make(blue_threshold_present_img)
cv2.imshow("hh",mask_present_img2)
cv2.waitKey(0)
mask_present_img2 = cv2.resize(mask_present_img2, img.shape[1::-1])
mask_present_img2 = cv2.cvtColor(mask_present_img2, cv2.COLOR_GRAY2RGB)
print(mask_present_img2.shape)
dst = cv2.bitwise_and(img1,mask_present_img2)
cv2.imshow("hh",dst)
cv2.waitKey(0)
cv2.imwrite("mask_p.jpg",dst)
img = np.asarray(Image.open("./mask_p.jpg").convert("L")).reshape(-1,1)
fig = plt.figure()
plt.hist(img, bins=128)
plt.show()
fig.savefig("hist.jpg")