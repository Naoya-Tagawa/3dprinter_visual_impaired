l = [
    "plejjjGq_box_0.15mm ",
    "lejjjGq_box_0.15mm_ ",
    "ejjjGq_box_0.15mm_P ",
    "jjjGq_box_0.15mm_Pl ",
    "jGq_box_0.15mm_PIA- ",
    "Gq_box_0Ll5mm_PLA_M ",
    "x_0.l5mm_PLA_MK3_17 ",
]


# 投票テーブルを作成
vote_table = []




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
        #diff_length = len(before_text) - len(text)
        text_parse = before_text[i:]
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
            #max_text = before_text[0:i] + text_parse + text[len(text) - i + diff_length :]
            max_text = before_text[0:i] + text
            #new_text = text_parse + text[len(text) - i :]
            max_last_stop_id = len(max_text)
            
        #print(count)
        count = 0
    
    
    parts_text_step = 0
    for step in range(max_stop_id, max_last_stop_id):
        parts_text = max_text[step]
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





