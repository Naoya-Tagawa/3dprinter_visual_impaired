l = [
    "plejjjGq_box_0.15mm ",
    "lejjjGq_box_0.15mm_ ",
    "ejjjGq_box_0.15mm_P ",
    "jjjGq_box_0.15mm_Pl ",
    "jGq_box_0.15mm_PLA_ ",
    "Gq_box_0Ll5mm_PLA_M ",
    "x_0.l5mm_PLA_MK3_17 ",
]

data_list = []


# 投票テーブルを作成
vote_table = {}


count_max = 0
max_text = ""
count = 0
text1 = "plejjjGq_box_0.15mm "
text2 = "lejjjGq_box_0.15mm_ "


for i in range(len(text1)):
    text1 = text1.strip()
    text2 = text2.strip()
    text_parse = text1[i:]
    print(text_parse)
    for j in range(len(text_parse)):
        if text_parse[j] == text2[j]:
            count += 2
        else:
            count += 0
    count += i
    if count_max < count:
        count_max = count
        max_text = text1[0:i] + text_parse + text2[len(text2) - i :]
    print(count)
    count = 0
print(max_text)
print(count_max)
print(text2[-1])
