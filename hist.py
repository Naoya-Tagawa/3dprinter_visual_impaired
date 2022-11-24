import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
dst = cv2.bitwise_and(img1,img1,mask=mask_present_img2)
cv2.imshow("hh",dst)
cv2.waitKey(0)
cv2.imwrite("mask_p.jpg",dst)

present_char_List1 , mask_present_img2 = img_processing2.mask_make(blue_threshold_present_img)
hist_mask = cv2.calcHist([img],[0],mask_present_img2,[256],[0,256])
color = ('b','g','r')
img1[img1 >= 255] = 0
dst = cv2.imread("./mask_p.jpg")
target_color = (255, 255, 255)

# 変更後の色
change_color = (0, 0, 0)

# 画像の縦横
h, w = img.shape[:2]

# 色の変更
for i in range(h):
    for j in range(w):
        b, g, r = img[i, j]
        if (b, g, r) == target_color:
            img[i, j] = change_color
#print(count)
#dst = cv2.bitwise_and(img,img,mask=mask_present_img2)
cv2.imshow("hh",img1)
cv2.waitKey(0)



histr1 = cv2.calcHist([img],[0],mask_present_img2,[256],[0,256])
histr2 = cv2.calcHist([img],[1],mask_present_img2,[256],[0,256])
histr3 = cv2.calcHist([img],[2],mask_present_img2,[256],[0,256])
# Figureを追加
fig = plt.figure(figsize = (8, 8))

# 3DAxesを追加
ax = fig.add_subplot(111, projection='3d')

# Axesのタイトルを設定
ax.set_title("", size = 20)

# 軸ラベルを設定
ax.set_xlabel("x", size = 14, color = "r")
ax.set_ylabel("y", size = 14, color = "r")
ax.set_zlabel("z", size = 14, color = "r")

# 軸目盛を設定
ax.set_xticks(np.arange(0,256,step=100))
ax.set_yticks(np.arange(0,256,step=100))

# -5～5の乱数配列(100要素)
x = 10 * np.random.rand(100, 1) - 5
y = 10 * np.random.rand(100, 1) - 5
z = 10 * np.random.rand(100, 1) - 5

# 曲線を描画
ax.scatter(histr1, histr2, histr3, color = "blue")

plt.show()