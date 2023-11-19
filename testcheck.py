def check_last_five_elements(lst):
    # リストの長さが5未満の場合は条件を満たすことができません
    if len(lst) < 5:
        return False

    # リストの後ろから5つの要素が全て1であるかを判定
    last_five_elements_are_ones = all(x == 1 for x in lst[-5:])

    # リスト全体から1の要素の数を数える
    count_of_ones = lst.count(1)

    # 条件を満たす要素が5つ以上あるかを判定
    return last_five_elements_are_ones and count_of_ones >= 5


k = [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(check_last_five_elements(k))
