import cv2
import numpy as np
import os
import glob

# チェスボード画像から算出したカメラパラメータを設定
fx = 559.3006089
fy = 559.46292818
Cx = 309.64749103
Cy = 228.93980934
mtx = np.array([[fx, 0, Cx],[0, fy, Cy],[0, 0, 1]])

# チェスボード画像から算出した歪係数を設定
k1 = -0.02672116
k2 = 0.09299264
p1 = -0.00760871
p2 = -0.00255127
k3 = -0.61769096
dist = np.array([[k1, k2, p1, p2, k3]])


# img_resizedフォルダー内のjpg画像を読み込む
images = glob.glob('./dataset_make/calibration/testImage/*.jpg')
print(images)
# Using the derived camera parameters to undistort the image
for filepath in images:

    img = cv2.imread(filepath)
    h,w = img.shape[:2]
    # Refining the camera matrix using parameters obtained by calibration
    # ROI:Region Of Interest(対象領域)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Method 1 to undistort the image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # undistort関数と同じ結果が返されるので、今回はコメントアウト(initUndistortRectifyMap()関数)
    # Method 2 to undistort the image
    # mapx,mapy=cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    # dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

    # 歪み補正した画像をimg_undistortフォルダへ保存
    cv2.imwrite('./dataset_make/calibration/img_undistort/distort_' + str(os.path.basename(filepath)), dst)
    cv2.waitKey(0)