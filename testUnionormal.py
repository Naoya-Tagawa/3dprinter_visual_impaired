import cv2
import numpy as np
import multiprocessing
from sklearn.neighbors import NearestNeighbors
from MakeVoicefile.VoiceProcessing import text_read
from CharacterRecog.CharacterRecog import load_model, TextRecog
import time
from ImageProcessing.img_processing2 import (
    sabun1,
    mask_make1,
    make_char_list,
    get_unique_list,
    mask_make,
    cut_blue_img1,
    cut_blue_img2,
    text_union,
    check_last_five_elements,
    diff_image_search_first,
    find_nearest_index,
)

# flag = True: 音声出力
# flag = false: 音声出力しない


def diff_image_search(
    present_frame,
    before_frame,
    before_frame_row1,
    before_frame_row2,
    before_frame_row3,
    before_frame_row4,
    output_text,
    model_pca,
    scaler,
    pca,
):
    img = cv2.imread("./MaskBlack/balck_img.jpg")
    output_union = [[], [], [], []]
    last_insert_time = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    model = cv2.createBackgroundSubtractorMOG2(history=3, detectShadows=False)
    flg_count = []
    text_union_index = 0
    before_text_union_index = 0
    result = ""
    while True:
        if len(flg_count) > 15:
            flg_count = []
        bad_accuracy_flg = []
        frame = present_frame.get()
        last_insert_time = time.time()
        text_union_output = ""
        blue_threshold_present_img = cut_blue_img2(frame)
        before_frame_row = []
        sabun_count = 0
        present_char_List2, mask_present_img2 = mask_make(blue_threshold_present_img)
        mask_frame = mask_present_img2.copy()
        before_frame = before_frame.astype("float")
        l2 = len(present_char_List2)

        if l2 > 4:
            blue_threshold_present_img = cut_blue_img1(frame)
            mask_frame = mask_make1(blue_threshold_present_img)
            # blue = cut_blue_trans2(present_frame)
            mask = model.apply(mask_frame)
            # cv2.accumulateWeighted(mask_present_img2, before_frame, 0.8)
            # frame_diff = mask_present_img2 - cv2.convertScaleAbs(before_frame)

            # frame_diff[frame_diff ==205] = 0
            # frame_diff = cv2.absdiff(mask_present_img2,cv2.convertScaleAbs(before_frame))
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.erode(mask, kernel, iterations=1)
            # mask = cv2.dilate(mask, kernel, iterations=1)
            # frame_diff = cv2.dilate(frame_diff,kernel)
            # cv2.imwrite("./ProcessingDisplay/raaa.jpg",frame_diff)
        else:
            # blue = cut_blue_trans(present_frame)
            mask = model.apply(mask_frame)
            # cv2.accumulateWeighted(mask_present_img2, before_frame, 0.8)
            # frame_diff = mask_present_img2 - cv2.convertScaleAbs(before_frame)
            # frame_diff[frame_diff == 205] = 0
            # frame_diff = cv2.absdiff(mask_present_img2,cv2.convertScaleAbs(before_frame))
            mask = cv2.erode(mask, kernel, iterations=1)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # cv2.imwrite("raaa.jpg",frame_diff)
        mask_frame[mask == 0] = 0
        frame_diff = cv2.morphologyEx(mask_frame, cv2.MORPH_OPEN, kernel)

        # plt.imshow(mask_present_img2)
        # plt.show()
        # h ,w = present_frame.shape
        # print(before_frame_row.shape)

        # before_frame = cv2.resize(before_frame,dsize=(w,h))
        contours, hierarchy = cv2.findContours(frame_diff.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 10:
                frame_diff = cv2.fillPoly(frame_diff, [contours[i][:, 0, :]], (0, 255, 0), lineType=cv2.LINE_8, shift=0)
        # cv2.imwrite("./ProcessingDisplay/realtimeimg_{0}.jpg".format(last_insert_time), frame_diff)
        cv2.imwrite("./ProcessingDisplay/mask_frame_{0}.jpg".format(last_insert_time), mask_present_img2)
        # plt.imshow(frame_diff)
        cv2.imwrite("framediff.jpg", frame_diff)
        # cv2.imshow("before.jpg",before_frame)

        # cv2.imshow("mas",mask_present_img2)
        # plt.show()
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        present_char_List1 = make_char_list(frame_diff)
        # フレームの差違のマスクの文字の位置を取得
        l1 = len(present_char_List1)
        # try:

        # cv2.imshow("gg",blue)
        # cv2.waitKey(0)
        # p1,p2,p3,p4 = points_extract2(blue)
        # except TypeError:
        # print("Screen cannot be detected")

        # syaei_img,M = projective_transformation2(mask_present_img2,p1,p2,p3,p4)

        # print(present_char_List1)
        # List = [ [0,y] for l in present_char_List1 for y in l]
        # present_char_List2 = make_char_list(mask_present_img2)
        # print(present_char_List2)
        # List = [ [[0,y] for y in l ]for l in present_char_List1]
        # print(List)
        # pt = cv2.perspectiveTransform(np.array([List]),M)
        if l1 != 0:
            try:
                knn_model = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(present_char_List2)
                distances, indices = knn_model.kneighbors(present_char_List1)
                indices = get_unique_list(indices)
                flg_count.append(0)
            except ValueError:
                indices = []
                # 認識するものがないことを示す
                flg_count.append(1)
        else:
            indices = []
            # 認識するものがないことを示す
            flg_count.append(1)
        # print(present_char_List2)

        # print(indices)
        # nearest_indexes = [find_nearest_index(present_char_List2, value) for value in indices]
        # print(nearest_indexes)
        for value in indices:
            if len(indices) == 0:
                break
            elif len(indices) > 4:
                break
            cut_present = mask_present_img2[
                int(present_char_List2[value[0]][0]) : int(present_char_List2[value[0]][1]),
            ]
            if l2 == l1:
                before_frame_row.append(cut_present)
                # flgが1なら前と

            # if arrow_exist(cut_present):
            # cut_present,judge = arrow_exist_judge(cut_present)
            # cv2.imshow("HHH",cut_present)
            # cv2.waitKey(0)
            # before_frame_row.append(cut_present)
            if not sabun1(before_frame_row1, cut_present, l1):
                sabun_count += 1

            if not sabun1(before_frame_row2, cut_present, l1):
                sabun_count += 1

            if not sabun1(before_frame_row3, cut_present, l1):
                sabun_count += 1

            if not sabun1(before_frame_row4, cut_present, l1):
                sabun_count += 1

            # cut_present1 = mask_present_img[int(j[0]):int(j[1]),]
            # print(sabun_count)
            if sabun_count > 3:
                out, accuracy = TextRecog(model_pca, scaler, pca, cut_present)
                if accuracy < 0.9:
                    bad_accuracy_flg.append(False)
                    print("破棄:")
                    print(out)
                    continue

                # out = recog_text(cut_present)
                # 矢印があるかどうか判定
                if out[0:1] == ">":
                    sabun_count = 0
                    print("精度高い:")
                    print(out)
                    out = out.split(">")
                    text_union_index = value[0]
                    if out[1] != "":
                        text_union_output = out[1]

                    print(text_union_output)
                else:
                    output_union[value[0]].append(out)

                    # output_textx = output_textx + " The cursor points to "
                # output_textx = output_textx + " \n" + out
            # before_frame_row.append(cut_present)
            # try:
            # if not sabun(before_arrow,cut_present):
            # output_text_p,out = img_processing2.match_text2(img_temp,label_temp,cut_present1)
            # if out != "":
            # output_textx.append(out)
            # except UnboundLocalError:
            # output_text_p,out = img_processing2.match_text2(img_temp,label_temp,cut_present1)
            # if out != "":
            # output_textx.append(out)
            # 矢印があるかどうか判定
            # if arrow_exist(cut_present):
            # before_frame_row.append(cut_present)
            sabun_count = 0
        if all(bad_accuracy_flg) == False:
            mask = model.apply(before_frame)
            continue

        if (check_last_five_elements(flg_count) == False) and (text_union_output != ""):
            output_union[text_union_index].append(text_union_output)
            print("output_union:" + str(output_union))
            for v in present_char_List2:
                if l2 == 0:
                    break
                elif l2 > 4:
                    break
                cut_present = mask_present_img2[int(v[0]) : int(v[1]),]
                before_frame_row.append(cut_present)
            out = ""
        elif (check_last_five_elements(flg_count) == True) and (output_union != [[], [], [], []]):
            print("output_union:" + str(output_union))
            for index, text_list in enumerate(output_union):
                if index == text_union_index:
                    output = "The cursor points to " + text_union(text_list)
                else:
                    try:
                        output = text_list[0]
                    except IndexError:
                        output = ""
                result = result + output + "\n"
            output_text.put(result)
            result = ""
            flg_count = []
            output_union = [[], [], [], []]

        # start1 = time.perf_counter()
        # end1 = time.perf_counter()
        # mask_present_img2,judge = arrow_exist_judge(mask_present_img2)
        try:
            if len(present_char_List2) == 0:
                before_frame_row1 = img
                before_frame_row2 = img
                before_frame_row3 = img
                before_frame_row4 = img
                before_frame = mask_present_img2
            elif len(present_char_List2) == 1:
                before_frame_row1 = before_frame_row[0]
                before_frame_row2 = img
                before_frame_row3 = img
                before_frame_row4 = img
                before_frame = mask_present_img2

            elif len(present_char_List2) == 2:
                before_frame_row1 = before_frame_row[0]
                before_frame_row2 = before_frame_row[1]
                before_frame_row3 = img
                before_frame_row4 = img
                before_frame = mask_present_img2

            elif len(present_char_List2) == 3:
                before_frame_row1 = before_frame_row[0]
                before_frame_row2 = before_frame_row[1]
                before_frame_row3 = before_frame_row[2]
                before_frame_row4 = img
                before_frame = mask_present_img2

            elif len(present_char_List2) == 4:
                before_frame_row1 = before_frame_row[0]
                before_frame_row2 = before_frame_row[1]
                before_frame_row3 = before_frame_row[2]
                before_frame_row4 = before_frame_row[3]
                before_frame = mask_present_img2

            else:
                before_frame_row1 = img
                before_frame_row2 = img
                before_frame_row3 = img
                before_frame_row4 = img
                before_frame = mask_present_img2
        except IndexError:
            before_frame_row1 = img
            before_frame_row2 = img
            before_frame_row3 = img
            before_frame_row4 = img
            before_frame = mask_present_img2
        cv2.imwrite("before_frame_row1.jpg", before_frame_row1)
        cv2.imwrite("before_frame_row2.jpg", before_frame_row2)
        cv2.imwrite("before_frame_row3.jpg", before_frame_row3)
        cv2.imwrite("before_frame_row4.jpg", before_frame_row4)


if __name__ == "__main__":
    # テンプレートをロード
    model, scaler, pca = load_model()
    cap = cv2.VideoCapture(0)
    read_fps = cap.get(cv2.CAP_PROP_FPS)
    print(read_fps)
    voice_flag = multiprocessing.Value("i", 0)
    # voice_flagが1なら今発話中,0なら発話していない
    count = 0
    # text_img = multiprocessing.Queue()
    output_text = multiprocessing.Queue()
    read = multiprocessing.Process(target=text_read, args=(output_text,))
    read.start()

    # 最初のフレームを取得する
    ret, bg = cap.read()
    before_frame_row1, before_frame_row2, before_frame_row3, before_frame_row4, before_frame = diff_image_search_first(
        bg
    )
    frame = bg
    count = 0
    before = bg
    h, w = frame.shape[:2]
    base = np.zeros((h, w, 3), np.uint32)
    # before_frame = None
    present_frame = multiprocessing.Queue()
    image_deal = multiprocessing.Process(
        target=diff_image_search,
        args=(
            present_frame,
            before_frame,
            before_frame_row1,
            before_frame_row2,
            before_frame_row3,
            before_frame_row4,
            output_text,
            model,
            scaler,
            pca,
        ),
    )
    image_deal.start()
    while True:
        ret, frame = cap.read()
        # フレームが取得できない場合は画面を閉じる
        if not ret:
            cv2.destroyAllWindows()
        cv2.imshow("frame", frame)
        # 画面が遷移したか調査
        # dst1 = cv2.bitwise_and(before,before,mask=before_frame)
        # dst2 = cv2.bitwise_and(frame,frame,mask=before_frame)
        # cv2.imshow("ll",dst2)
        # dst1[dst1 >= 255] = 0
        # dst2[dst2>= 255] = 0
        # h,w,e = dst1.shape
        # print(h*w)
        # count1 =  sum(((r>0) and (g>0) and (b>0)) for d in dst1 for r,g,b in d)
        # count2 =  sum(((r>0) and (g>0) and (b>0)) for d in dst2 for r,g,b in d)
        # dst0 = list(itertools.chain.from_iterable(dst1))
        # dst3 = list(itertools.chain.from_iterable(dst2))
        # dst1_count = sum(((b>0) and (r>150)) for b,g,r in dst0)
        # dst2_count = sum(((b>0) and (r>150)) for b,g,r in dst3)
        # per = (dst2_count / dst1_count) * 100
        # print(dst1_count)
        # print(dst2_count)
        if count == 2:
            base = frame + base
            base = base / 3
            base = base.astype(np.uint8)
            # cv2.imwrite("base17.jpg",base)
            present_frame.put(base)
            # before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4,before_frame= diff_image_search(base,before_frame,before_frame_row1,before_frame_row2,before_frame_row3,before_frame_row4,output_text,img_temp,label_temp)
            count = 0
            base = np.zeros((h, w, 3), np.uint32)

        else:
            base = base + frame
            count += 1
            # time.sleep(0.04)
        before = frame

        # diff_flag = Trueなら画面遷移,diff_flag=Falseなら画面遷移していない
        # present_kersol = audio_output.kersol_search(output_text)
        # if present_kersol == 1: # カーソルがない
        # qキーが入力されたら画面を閉じる
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break

    read.join()
