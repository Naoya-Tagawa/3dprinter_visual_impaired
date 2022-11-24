import cv2
import img_processing2

img = cv2.imread("./hei/camera186.jpg")
img1 = cv2.imread("./hei/camera181.jpg")
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
h,w,d = img1.shape
    #フレームの青い部分を二値化
img1 = cv2.resize(img1, img.shape[1::-1])
blue_threshold_present_img = img_processing2.cut_blue_img2(img1)


present_char_List1 , mask_present_img2 = img_processing2.mask_make(blue_threshold_present_img)
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
<<<<<<< HEAD
cv2.imwrite("mask_p.jpg",dst)
=======
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



for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],mask_present_img2,[256],[0,256])
    histr += histr
plt.plot(histr,color = 'g')
plt.xlim([0,256])



plt.savefig("hist6.png")
plt.show()


>>>>>>> d678f219b78c309e7e939ea2e8f7ceff7e14e969
