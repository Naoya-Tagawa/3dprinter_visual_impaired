import matplotlib.pyplot as plt
import cv2
import numpy as np

import cv2
import numpy as np

# 無処理
def nothing(x):
    pass

# ウィンドウの生成
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# トラックバーの生成
cv2.createTrackbar('minH', 'image', 0, 255, nothing)
cv2.createTrackbar('maxH', 'image', 255, 255, nothing)
cv2.createTrackbar('minS', 'image', 0, 255, nothing)
cv2.createTrackbar('maxS', 'image', 255, 255, nothing)
cv2.createTrackbar('minV', 'image', 0, 255, nothing)
cv2.createTrackbar('maxV', 'image', 255, 255, nothing)

# イメージの読み込み
img = cv2.imread('./blue_threshold_present.jpg')

# HSV変換
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

while True:
    # トラックバーの値の取得
    minH = cv2.getTrackbarPos('minH', 'image')
    minS = cv2.getTrackbarPos('minS', 'image')
    minV = cv2.getTrackbarPos('minV', 'image')
    maxH = cv2.getTrackbarPos('maxH', 'image')
    maxS = cv2.getTrackbarPos('maxS', 'image')
    maxV = cv2.getTrackbarPos('maxV', 'image')

    # 画像の更新
    img_mask = cv2.inRange(img_hsv, np.array([minH, minS, minV]), np.array([maxH, maxS, maxV]))
    cv2.imshow('image', img_mask)

    # qキーで終了
    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

# 破棄
cv2.destroyAllWindows()






# 前景画像を読み込む。
fg_img = cv2.imread("sample2.jpg")

# 背景画像を読み込む。
bg_img = cv2.imread("sample3.jpg")

# HSV に変換する。
hsv = cv2.cvtColor(fg_img, cv2.COLOR_BGR2HSV)

# 2値化する。
bin_img = cv2.inRange(hsv, (0, 10, 0), (255, 255, 255))
plt.imshow(bin_img)
plt.show()
# 輪郭抽出する。
contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 面積が最大の輪郭を取得する
contour = max(contours, key=lambda x: cv2.contourArea(x))
# マスク画像を作成する。
mask = np.zeros_like(bin_img)
cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
plt.imshow(mask)
plt.show()
x, y = 10, 10  # 貼り付け位置

# 幅、高さは前景画像と背景画像の共通部分をとる
w = min(fg_img.shape[1], bg_img.shape[1] - x)
h = min(fg_img.shape[0], bg_img.shape[0] - y)

# 合成する領域
fg_roi = fg_img[:h, :w]  # 前傾画像のうち、合成する領域
bg_roi = bg_img[y : y + h, x : x + w]  # 背景画像のうち、合成する領域

plt.imshow(bg_roi[:])
plt.show()

# 合成する。
bg_roi[:] = np.where(mask[:h, :w, np.newaxis] == 0, bg_roi, fg_roi)
plt.imshow(bg_img)
plt.show()