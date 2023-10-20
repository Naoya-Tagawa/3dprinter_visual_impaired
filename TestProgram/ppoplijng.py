import cv2

# 画像の読み込み
img = cv2.imread('mask_p.jpg')

# 指定色
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




color = ('b','g','r')
cv2.imshow("ee",img)
cv2.waitKey(0)
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],mask_present_img2,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig("hist2.png")
plt.show()