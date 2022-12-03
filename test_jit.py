import cv2
import img_processing2

img = cv2.imread("./hei/camera186.jpg")
img1 = cv2.imread("./hei/camera181.jpg")
img2 = cv2.imread("./hei/camer120.jpg")
img3 = cv2.imread("./hei/camera618.jpg")
img4 = cv2.imread("./hei/camera518.jpg")
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
h,w,d = img1.shape
    #フレームの青い部分を二値化
img1 = cv2.resize(img1, img.shape[1::-1])
blue_threshold_present_img = img_processing2.cut_blue_img2(img1)


present_char_List1 , mask_present_img2 = img_processing2.mask_make(blue_threshold_present_img)

blue_threshold_present_img7 = img_processing2.cut_blue_img2(img4)
present_char_List1 , mask8 = img_processing2.mask_make(blue_threshold_present_img7)

cv2.imshow("hh",img)
cv2.waitKey(0)
#mask_present_img2 = cv2.resize(mask_present_img2, img.shape[1::-1])
#mask_present_img2 = cv2.cvtColor(mask_present_img2, cv2.COLOR_GRAY2BGR)
blue_threshold_present_img1 = img_processing2.cut_blue_img2(img)
present_char_List1 , mask_present_img3 = img_processing2.mask_make(blue_threshold_present_img1)
print(mask_present_img2.shape)
dst = cv2.bitwise_and(img,mask_present_img2)
cv2.imshow("hh",dst)
cv2.waitKey(0)
cv2.imwrite("mask_p.jpg",dst)
=======
cv2.imwrite("mask_p.jpg",dst)

present_char_List1 , mask_present_img2 = img_processing2.mask_make(blue_threshold_present_img)
hist_mask = cv2.calcHist([img],[0],mask_present_img2,[256],[0,256])
color = ('b','g','r')
img2[img2 >= 255] = 0
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
dst = cv2.bitwise_and(img2,img2,mask=mask_present_img2)
cv2.imshow("hh",dst)
cv2.waitKey(0)
cv2.imwrite("dst.jpg",dst)

dst7 = cv2.bitwise_and(img3,img3,mask=mask8)
#dst[dst >= 255] = 0
cv2.imshow("dst7",dst7)
cv2.waitKey(0)
cv2.imwrite("dst.jpg",dst)

for i,col in enumerate(color):
    histr = cv2.calcHist([img2],[i],mask_present_img2,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    #histr += histr




plt.savefig("hist8.png")
plt.show()


