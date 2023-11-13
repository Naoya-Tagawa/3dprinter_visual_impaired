import cv2
camera = cv2.VideoCapture(0)                               # カメラCh.(ここでは0)を指定
camera1 = cv2.VideoCapture(2)
# 動画ファイル保存用の設定
fps = int(camera.get(cv2.CAP_PROP_FPS))                    # カメラのFPSを取得
w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))              # カメラの横幅を取得
h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))             # カメラの縦幅を取得
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
video = cv2.VideoWriter('C:\\Users\\naoya\\Desktop\\video2.wave', fourcc, fps, (w, h))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）
# 動画ファイル保存用の設定
fps1 = int(camera1.get(cv2.CAP_PROP_FPS))                    # カメラのFPSを取得
w1 = int(camera1.get(cv2.CAP_PROP_FRAME_WIDTH))              # カメラの横幅を取得
h1 = int(camera1.get(cv2.CAP_PROP_FRAME_HEIGHT))             # カメラの縦幅を取得
fourcc1 = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
video1 = cv2.VideoWriter('C:\\Users\\naoya\\Desktop\\video3.wave', fourcc1, fps1, (w1, h1))
# 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
while True:
    ret, frame = camera.read()
    ret, frame1 = camera1.read()                              # フレームを取得
    cv2.imshow('camera', frame)                          # フレームを画面に表示
    cv2.imshow('camera1', frame1)
    video.write(frame)
    video1.write(frame1)                                    # 動画を1フレームずつ保存する
    # キー操作があればwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 撮影用オブジェクトとウィンドウの解放
camera.release()
camera1.release()
cv2.destroyAllWindows()