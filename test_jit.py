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
blue_threshold_present_img1 = img_processing2.cut_blue_img2(img)


present_char_List1 , mask_present_img3 = img_processing2.mask_make(blue_threshold_present_img1)


cv2.imshow("hh",img)
cv2.waitKey(0)
mask_present_img2 = cv2.resize(mask_present_img2, img.shape[1::-1])
mask_present_img2 = cv2.cvtColor(mask_present_img2, cv2.COLOR_GRAY2RGB)
print(mask_present_img2.shape)
dst = cv2.bitwise_and(img,mask_present_img2)
cv2.imshow("hh",dst)
cv2.waitKey(0)
cv2.imwrite("mask_p.jpg",dst)
cv2.imshow("hh",mask_present_img3)
cv2.waitKey(0)
print(present_char_List1)
for i in present_char_List1:
    if len(present_char_List1)==0:
        break
    elif len(present_char_List1) > 4:
        break
    cut_present = mask_present_img3[int(i[0]):int(i[1]),]
    cv2.imshow("kk",cut_present)
    cv2.waitKey(0)