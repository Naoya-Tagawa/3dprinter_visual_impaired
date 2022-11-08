import numpy as np
import cv2
import os
from PIL import Image

#画像の読み込み
#img = cv2.imread("./camera1/camera39.jpg")
img=cv2.imread("./mask.png")
kernel = np.ones((5,5),np.uint8)
mask_present_img2 = cv2.dilate(img,kernel,iterations=1)
cv2.imshow('after gamma',mask_present_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


#γ変換の値
gamma=0.1
#γ変換の対応表を作る
LUT_Table=np.zeros((256,1),dtype='uint8')
print(len(LUT_Table))
for i in range(len(LUT_Table)):
    LUT_Table[i][0]=255*(float(i)/255)**(1.0/gamma)

#γ変換をする
img_gamma=cv2.LUT(img,LUT_Table)
cv2.imwrite("gammma.jpg",img_gamma)
#画像の表示
cv2.imshow('original',img)
cv2.imshow('after gamma',img_gamma)
cv2.waitKey(0)
cv2.destroyAllWindows()


