import numpy as np


def find_nearest_index(array, target):
    array = np.asarray(array)
    target = np.asarray(target)

    # 距離の計算
    distances = np.linalg.norm(array - target, axis=1)

    # 最小距離のインデックスを返す
    nearest_index = np.argmin(distances)

    return nearest_index


# 入力データ
data = np.array([[253.0, 288.0], [292.0, 328.0], [332.0, 368.0], [371.0, 409.0]])

# ターゲット値
target_value = np.array([292.0, 328.0])

# 最も近い行のインデックスを取得
result_index = find_nearest_index(data, target_value)

print("最も近い行のインデックス:", result_index)
