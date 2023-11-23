l = [
    "jjGq_box_0.11m_PLA ",
    "jjGq_box_0.11m_PLA ",
    "jjGq_box_0.11m_PLA ",
    "jjGq_box_0.11m_PLA ",
    "GqabbexL015LLPlLuL ",
    "GqabbexL015LLPlLuL ",
    "GqabbexL015LLPlLuL ",
    "GqabbexL015LLPlLuL ",
    "q_box_G.15m_PLA_MK ",
    "q_box_G.15m_PLA_MK ",
    "q_box_G.15m_PLA_MK ",
    "q_box_G.15m_PLA_MK ",
    "_box_0.1LPLA_MK3 ",
    "_box_0.1LPLA_MK3 ",
    "_box_0.1LPLA_MK3 ",
    "_box_0.1LPLA_MK3 ",
    "box_0.15vLA_MK3_ ",
    "box_0.15vLA_MK3_ ",
    "box_0.15vLA_MK3_ ",
    "box_0.15vLA_MK3_ ",
]


def text_union(l):
    # 投票テーブルを作成
    vote_table = []
    # 類似度の最大値をもつイテレータ
    count_max = 0
    # 類似度が最大のテキスト
    max_text = ""
    # カウント
    count = 0
    max_stop_id = 0
    max_last_stop_id = 0
    before_text1 = ""

    for step, text in enumerate(l):
        text = text.strip()
        if step == 0:
            # 投票テーブルの更新
            vote_table = [{char: 1} for char in text]
            before_text = text
            continue
        if before_text1 == text:
            continue

        for i in range(len(text)):
            before_text = before_text.strip()
            # diff_length = len(before_text) - len(text)
            text_parse = before_text[i:]
            # print(text_parse)
            # print(text)
            for j in range(len(text_parse)):
                if j >= len(text):
                    break
                else:
                    if text_parse[j] == text[j]:
                        count += 2
                    else:
                        count += 0

            # count += i
            # print(count)
            # print("\n")
            if count_max < count:
                count_max = count
                max_stop_id = i
                # max_text = before_text[0:i] + text_parse + text[len(text) - i + diff_length :]
                max_text = before_text[0:i] + text
                max_last_stop_id = len(max_text)
            count = 0

        for step in range(max_stop_id, max_last_stop_id):
            parts_text = max_text[step]
            if step >= len(vote_table):
                vote_table.append({parts_text: 1})
            else:
                if parts_text in vote_table[step].keys():
                    vote_table[step].update({parts_text: vote_table[step][parts_text] + 1})
                else:
                    vote_table[step].update({parts_text: 1})

        before_text = "".join(max(entry, key=entry.get) for entry in vote_table)
        count_max = 0
        before_text1 = text
        # print(vote_table)

    return "".join(max(entry, key=entry.get) for entry in vote_table), vote_table


result, table = text_union(l)
print(table)
print(result)
