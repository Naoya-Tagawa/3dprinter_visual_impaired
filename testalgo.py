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
vote_table = []


text0 = l[0]
for i in range(len(text0)):
    vote_table.append({text0[i]: 1})

#print(vote_table)
#vote_table[0].update({"a":1})
#print(vote_table)

count_max = 0
max_text = ""
count = 0
text1 = "plejjjGq_box_0.15mm "
text2 = "lejjjGq_box_0.15mm_ "
max_stop_id = 0
max_last_stop_id = 0
new_text = ""

for step ,text in enumerate(l):
    text = text.strip()
    if step == 0:
        #投票テーブルの更新
        vote_table = [{char: 1} for char in text]
        before_text = text
        continue
    
    for i in range(len(text)):
        before_text = before_text.strip()
        text_parse = before_text[i:]
        print(text_parse)
        print(text)
        for j in range(len(text_parse)):
            if j >= len(text):
                break
            else:
                if text_parse[j] == text[j]:
                    count += 2
                else:
                    count += 0
        count += i
        if count_max < count:
            count_max = count
            max_stop_id = i
            max_text = before_text[0:i] + text_parse + text[len(text) - i :]
            #new_text = text_parse + text[len(text) - i :]
            max_last_stop_id = len(max_text)
            
        #print(count)
        count = 0
    print(max_text)
    
    parts_text_step = 0
    for step in range(max_stop_id, max_last_stop_id):
        parts_text = max_text[step]
        print(step)
        print(parts_text)
        #print(max_last_stop_id)
        #print(len(vote_table))
        if step >= len(vote_table):
            vote_table.append({parts_text: 1})
        else:
            if parts_text in vote_table[step].keys():
                vote_table[step].update({parts_text: vote_table[step][parts_text] + 1})
            else:
                vote_table[step].update({parts_text: 1})

        parts_text_step += 1
            
    before_text = ''.join(max(entry, key=entry.get) for entry in vote_table)
    #print(vote_table)
    count_max = 0
            
    
    
print(vote_table)
print(''.join(max(entry, key=entry.get) for entry in vote_table))


    # for i in range(len(text)):
    #         text = text.strip()
    #         text2 = text2.strip()
    #         text_parse = text1[i:]
    #         print(text_parse)
    #         for j in range(len(text_parse)):
    #             if text_parse[j] == text2[j]:
    #                 count += 2
    #             else:
    #                 count += 0
    #         count += i
    #         if count_max < count:
    #             count_max = count
    #             max_text = text1[0:i] + text_parse + text2[len(text2) - i :]
    #         print(count)
    #         count = 0
    # print(max_text)
    # print(count_max)
    # print(text2[-1])


