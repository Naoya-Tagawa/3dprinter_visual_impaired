#ライブラリのインポート
import cv2
import numpy as np
import glob

#情報収集
pics='./hei/*.jpg'
pic_list=glob.glob(pics)
img=cv2.imread('./hei/camera124.jpg')
img1=cv2.imread('./hei/camera138.jpg')
h,w=img.shape[:2]
base_array=np.zeros((h,w,3),np.uint32)

#平均画像の作成
base_array=base_array+img+img1
base_array=base_array/2
print(len(pics))
base_array=base_array.astype(np.uint8)
cv2.imwrite('avg.jpg',base_array)