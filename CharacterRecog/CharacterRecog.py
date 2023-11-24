import pickle
import cv2
import numpy as np
import warnings
import time

warnings.simplefilter("ignore")


# 横方向のProjection profileを得る
def Projection_V(img, h, w):
    array_V = np.full(w, h)
    count = [np.count_nonzero(img[:, i : i + 1]) for i in range(w)]
    count = np.asarray(count, dtype=int)
    array_V = array_V - count
    return array_V


# Projection profileから横方向の座標を得る
def Detect_WidthPosition(W_THRESH, width, array_V):
    char_List = np.array([])

    flg = False
    posi1 = 0
    posi2 = 0
    for i in range(width):
        val = array_V[i]
        if flg == False and val < W_THRESH:
            flg = True
            posi1 = i

        if flg == True and val >= W_THRESH:
            flg = False
            posi2 = i
            char_List = np.append(char_List, posi1)
            char_List = np.append(char_List, posi2)

    return char_List


def preprocess_new_image(img):
    pixel_values_list = []
    # グレースケール化
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # サイズ統一
    # image = cv2.resize(img, (26, 36))
    # 画像のピクセル値を特徴量として1次元配列に変換
    pixels = img.flatten()
    pixel_values_list.append(pixels)
    return pixel_values_list


def load_model():
    with open("CharacterModel/3dprinter_model.pickle", mode="rb") as fp:
        model = pickle.load(fp)
    with open("CharacterModel/3dprinter_scaler.pickle", mode="rb") as fp:
        scaler = pickle.load(fp)
    with open("CharacterModel/3dprinter_pca.pickle", mode="rb") as fp:
        pca = pickle.load(fp)
    return model, scaler, pca


def load_randomforest_model():
    with open("CharacterModel/3dprinter_randomforest_model.pickle", mode="rb") as fp:
        model = pickle.load(fp)
    with open("CharacterModel/3dprinter_randomforest_scaler.pickle", mode="rb") as fp:
        scaler = pickle.load(fp)
    with open("CharacterModel/3dprinter_randomforest_pca.pickle", mode="rb") as fp:
        pca = pickle.load(fp)

    return model, scaler, pca


def predict(model, scaler, pca, img):
    image = preprocess_new_image(img)
    image_scaled = scaler.transform(image)
    image_pc = pca.transform(image_scaled)
    prediction = model.predict(image_pc[-1].reshape(1, -1))
    prediction = str(prediction[0])
    # print(prediction)
    # 各クラスの確率を取得
    class_probabilities = model.predict_proba(image_pc[-1].reshape(1, -1))

    # 最大確率のインデックスを取得
    max_prob_index = np.argmax(class_probabilities)

    # 最大確率とそのクラスを出力
    max_prob = class_probabilities[0, max_prob_index]
    # print("最大確率:", max_prob)
    return prediction, max_prob


def randomforest_predict2(model, scaler, img):
    img = preprocess_new_image(img)
    img_scaled = scaler.transform(img)
    prediction = model.predict(img_scaled[-1].reshape(1, -1))
    prediction = str(prediction[0])
    print(prediction)
    # 各クラスの確率を取得
    class_probabilities = model.predict_proba(img_scaled[-1].reshape(1, -1))

    # 最大確率のインデックスを取得
    max_prob_index = np.argmax(class_probabilities)

    # 最大確率とそのクラスを出力
    max_prob = class_probabilities[0, max_prob_index]
    print("最大確率:", max_prob)
    return prediction


def randomforest_predict(model, scaler, pca, img):
    image = preprocess_new_image(img)
    image_scaled = scaler.transform(image)
    image_pc = pca.transform(image_scaled)
    prediction = model.predict(image_pc[-1].reshape(1, -1))
    prediction = str(prediction[0])
    print(prediction)
    # 各クラスの確率を取得
    class_probabilities = model.predict_proba(image_pc[-1].reshape(1, -1))

    # 最大確率のインデックスを取得
    max_prob_index = np.argmax(class_probabilities)

    # 最大確率とそのクラスを出力
    max_prob = class_probabilities[0, max_prob_index]
    print("最大確率:", max_prob)
    return prediction


def RandomForestTextRecog2(model, scaler, frame):
    # 対象画像をリサイズ
    # 対象画像をグレイスケール化
    # start = time.time()
    # gray_img = cv2.cvtColor(syaei_resize_img,cv2.COLOR_BGR2GRAY)
    # 二値画像へ
    # ret, img_mask = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)
    img_mask = frame
    # img_mask = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,-3)

    # ノイズ除去
    # img_mask = cv2.medianBlur(img_mask,3)
    # 膨張化
    # img_mask = cv2.dilate(img_mask,kernel)
    # 高さ、幅を保持
    kernel = np.ones((3, 3), np.uint8)
    height, width = img_mask.shape
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    # img_mask = cv2.erode(img_mask,np.ones((5,5),np.uint8),iterations=1)
    # cv2.imshow("ll",img_erode)
    # cv2.waitKey(0)
    out_modify = ""  # 修正したテキスト
    # 横方向のProjection Profileを得る
    array_V = Projection_V(img_mask, height, width)
    W_THRESH = max(array_V)
    char_List2 = Detect_WidthPosition(W_THRESH, width, array_V)
    out_modify = ""  # 修正したテキスト
    out = ""  # 読み取ったテキスト
    # print(char_List2)
    img_mask = cv2.dilate(img_mask, kernel, iterations=1)

    for j in range(0, len(char_List2) - 1, 2):
        # end_time = time.perf_counter()
        # print(end_time-start_time)
        # 一文字ずつ切り取る
        match_img = img_mask[:, int(char_List2[j]) - 1 : int(char_List2[j + 1]) + 1]
        # cv2.imwrite("match.jpg",match_img)
        try:
            match_img = cv2.resize(match_img, dsize=(26, 36))
            # cv2.imshow("kk", match_img)
            # cv2.waitKey(0)
        except cv2.error:
            return ""
        height_m, width_m = match_img.shape
        # cv2.imshow("kk",match_img)
        # cv2.waitKey(0)
        # cv2.imwrite("match{0}.jpg".format(j),match_img)

        prediction = randomforest_predict2(model, scaler, match_img)
        # print(prediction)
        if (j != 0) & (char_List2[j] > (width_m + char_List2[j - 1])):
            if (j + 1) == len(char_List2) - 1:
                out_modify = out_modify + " " + prediction
                out = out + out_modify + " "
                # output_text.append('\n')
                out_modify = ""
                continue
                # out_modify = speling.correct(out_modify)
                # out_modify += label_temp[new_d[0][1]]
            out_modify += " "
            # out = out + out_modify
            # output_text.append(' ')
            # output_text.append(out_modify)
            # print(out_modify)
            # out_modify = ""
        # 行の最後の時
        if (j + 1) == len(char_List2) - 1:
            out_modify = out_modify + prediction
            # out_modify = speling.correct(out_modify)
            out = out + out_modify + " "
            # output_text.append('\n')
            out_modify = ""
            continue
        # print(label_temp[new_d[0][1]])
        out_modify = out_modify + prediction
        # print(out_modify)
        continue
    # end = time.time()
    # print("処理時間:",end-start)
    # f = open("test_logi.txt",'a', encoding='UTF-8')
    # f.write( '認識結果'+ out + '\n')
    # f.write('実行時間'+ str(end-start) + '\n')
    # f.close()
    return out


def RandomForestTextRecog(model, scaler, pca, frame):
    # 対象画像をリサイズ
    # 対象画像をグレイスケール化
    # start = time.time()
    # gray_img = cv2.cvtColor(syaei_resize_img,cv2.COLOR_BGR2GRAY)
    # 二値画像へ
    # ret, img_mask = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)
    img_mask = frame
    # img_mask = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,-3)

    # ノイズ除去
    # img_mask = cv2.medianBlur(img_mask,3)
    # 膨張化
    # img_mask = cv2.dilate(img_mask,kernel)
    # 高さ、幅を保持
    kernel = np.ones((3, 3), np.uint8)
    height, width = img_mask.shape
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    # img_mask = cv2.erode(img_mask,np.ones((5,5),np.uint8),iterations=1)
    # cv2.imshow("ll",img_erode)
    # cv2.waitKey(0)
    out_modify = ""  # 修正したテキスト
    # 横方向のProjection Profileを得る
    array_V = Projection_V(img_mask, height, width)
    W_THRESH = max(array_V)
    char_List2 = Detect_WidthPosition(W_THRESH, width, array_V)
    out_modify = ""  # 修正したテキスト
    out = ""  # 読み取ったテキスト
    # print(char_List2)
    img_mask = cv2.dilate(img_mask, kernel, iterations=1)

    for j in range(0, len(char_List2) - 1, 2):
        # end_time = time.perf_counter()
        # print(end_time-start_time)
        # 一文字ずつ切り取る
        match_img = img_mask[:, int(char_List2[j]) - 1 : int(char_List2[j + 1]) + 1]
        # cv2.imwrite("match.jpg",match_img)
        try:
            match_img = cv2.resize(match_img, dsize=(26, 36))
            # cv2.imshow("kk", match_img)
            # cv2.waitKey(0)
        except cv2.error:
            return ""
        height_m, width_m = match_img.shape
        # cv2.imshow("kk",match_img)
        # cv2.waitKey(0)
        # cv2.imwrite("match{0}.jpg".format(j),match_img)

        prediction = randomforest_predict(model, scaler, pca, match_img)
        # print(prediction)
        if (j != 0) & (char_List2[j] > (width_m + char_List2[j - 1])):
            if (j + 1) == len(char_List2) - 1:
                out_modify = out_modify + " " + prediction
                out = out + out_modify + " "
                # output_text.append('\n')
                out_modify = ""
                continue
                # out_modify = speling.correct(out_modify)
                # out_modify += label_temp[new_d[0][1]]
            out_modify += " "
            # out = out + out_modify
            # output_text.append(' ')
            # output_text.append(out_modify)
            # print(out_modify)
            # out_modify = ""
        # 行の最後の時
        if (j + 1) == len(char_List2) - 1:
            out_modify = out_modify + prediction
            # out_modify = speling.correct(out_modify)
            out = out + out_modify + " "
            # output_text.append('\n')
            out_modify = ""
            continue
        # print(label_temp[new_d[0][1]])
        out_modify = out_modify + prediction
        # print(out_modify)
        continue
    # end = time.time()
    # print("処理時間:",end-start)
    # f = open("test_logi.txt",'a', encoding='UTF-8')
    # f.write( '認識結果'+ out + '\n')
    # f.write('実行時間'+ str(end-start) + '\n')
    # f.close()
    return out


def TextRecog(model, scaler, pca, frame):
    # 対象画像をリサイズ
    # 対象画像をグレイスケール化
    # start = time.time()
    # gray_img = cv2.cvtColor(syaei_resize_img,cv2.COLOR_BGR2GRAY)
    # 二値画像へ
    # ret, img_mask = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)
    img_mask = frame
    # img_mask = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,-3)
    sum_recog_accuracy = 0
    row_recog_accuracy = 0
    recog_count = 0
    # ノイズ除去
    # img_mask = cv2.medianBlur(img_mask,3)
    # 膨張化
    # img_mask = cv2.dilate(img_mask,kernel)
    # 高さ、幅を保持
    kernel = np.ones((3, 3), np.uint8)
    height, width = img_mask.shape
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    # img_mask = cv2.erode(img_mask,np.ones((5,5),np.uint8),iterations=1)
    # cv2.imshow("ll",img_erode)
    # cv2.waitKey(0)
    out_modify = ""  # 修正したテキスト
    # 横方向のProjection Profileを得る
    array_V = Projection_V(img_mask, height, width)
    W_THRESH = max(array_V)
    char_List2 = Detect_WidthPosition(W_THRESH, width, array_V)
    out_modify = ""  # 修正したテキスト
    out = ""  # 読み取ったテキスト
    # print(char_List2)
    img_mask = cv2.dilate(img_mask, kernel, iterations=1)

    for j in range(0, len(char_List2) - 1, 2):
        # end_time = time.perf_counter()
        # print(end_time-start_time)
        # 一文字ずつ切り取る
        match_img = img_mask[:, int(char_List2[j]) - 1 : int(char_List2[j + 1]) + 1]
        # cv2.imwrite("match.jpg",match_img)
        try:
            match_img = cv2.resize(match_img, dsize=(26, 36))
            # cv2.imshow("kk", match_img)
            # cv2.waitKey(0)
        except cv2.error:
            return ""
        height_m, width_m = match_img.shape
        # cv2.imshow("kk",match_img)
        # cv2.waitKey(0)
        # cv2.imwrite("match{0}.jpg".format(j),match_img)

        prediction, acurracy = predict(model, scaler, pca, match_img)
        sum_recog_accuracy += acurracy
        recog_count += 1
        # print(prediction)
        if (j != 0) & (char_List2[j] > (width_m + char_List2[j - 1])):
            if (j + 1) == len(char_List2) - 1:
                out_modify = out_modify + " " + prediction
                out = out + out_modify + " "
                # output_text.append('\n')
                out_modify = ""
                continue
                # out_modify = speling.correct(out_modify)
                # out_modify += label_temp[new_d[0][1]]
            out_modify += " "
            # out = out + out_modify
            # output_text.append(' ')
            # output_text.append(out_modify)
            # print(out_modify)
            # out_modify = ""
        # 行の最後の時
        if (j + 1) == len(char_List2) - 1:
            out_modify = out_modify + prediction
            # out_modify = speling.correct(out_modify)
            out = out + out_modify + " "
            # output_text.append('\n')
            out_modify = ""
            continue
        # print(label_temp[new_d[0][1]])
        out_modify = out_modify + prediction
        # print(out_modify)
        continue
    average_accuracy = sum_recog_accuracy / recog_count
    # end = time.time()
    # print("処理時間:",end-start)
    # f = open("test_logi.txt",'a', encoding='UTF-8')
    # f.write( '認識結果'+ out + '\n')
    # f.write('実行時間'+ str(end-start) + '\n')
    # f.close()
    return out, average_accuracy


# if __name__ == "__main__":
#     img = cv2.imread("before_frame_row1.jpg")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     with open("CharacterModel/3dprinter_randomforest_model2.pickle", mode="rb") as f:
#         model = pickle.load(f)
#     with open("CharacterModel/3dprinter_randomforest_scaler2.pickle", mode="rb") as fp:
#         scaler = pickle.load(fp)
#     start = time.time()
#     text = RandomForestTextRecog2(model, scaler, img)
#     print(text)
#     end = time.time()
#     print(end - start)

#     model, scaler, pca = load_randomforest_model()
#     start = time.time()
#     print(start)
#     text = RandomForestTextRecog(model, scaler, pca, img)
#     print(text)
#     end = time.time()
#     print(end)
#     print(end - start)
#     model, scaler, pca = load_model()
#     start = time.time()
#     print(start)
#     text, ave = TextRecog(model, scaler, pca, img)
#     print(text, ave)
#     end = time.time()
#     print(end)
#     print(end - start)
#     cv2.imshow("kk", img)
#     cv2.waitKey(0)
