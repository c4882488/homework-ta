# import numpy as np
#
# def split_data(data, n_splits):
#     # 計算 Q1, Q3 和 IQR
#     Q1 = np.percentile(data, 25)
#     Q3 = np.percentile(data, 75)
#     IQR = Q3 - Q1
#
#     # 識別離群值
#     outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
#
#     # 分離離群值和非離群值
#     data_outliers = data[outliers]
#     data_without_outliers = data[~outliers]
#
#     # 計算非離群值的最小值和最大值
#     min_val = np.min(data_without_outliers)
#     max_val = np.max(data_without_outliers)
#
#     # 使用 linspace 生成 n_splits 個均勻間隔的數字
#     splits = np.linspace(min_val, max_val, n_splits+1)
#
#     # 創建一個空列表來存儲分割後的數據
#     split_data = []
#
#     # 將非離群值分割成 n_splits 組
#     for i in range(n_splits):
#         if i == n_splits - 1:
#             split_data.append(data_without_outliers[(data_without_outliers >= splits[i]) & (data_without_outliers <= splits[i+1])])
#         else:
#             split_data.append(data_without_outliers[(data_without_outliers >= splits[i]) & (data_without_outliers < splits[i+1])])
#
#     # 將離群值加回到最前面或最後面的分組中
#     for outlier in data_outliers:
#         if outlier < min_val:
#             split_data[0] = np.append(split_data[0], outlier)
#         else:
#             split_data[-1] = np.append(split_data[-1], outlier)
#
#     # 返回分割後的數據
#     return split_data
#
# data = np.array([1, 100, 300, 20, 50, 4, -1])
# print(split_data(data, 4))


import numpy as np

def split_data(data, n_splits):
    # Calculate Q1, Q3, and IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Identify outliers
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))

    # Separate outliers and non-outliers
    data_outliers = data[outliers]
    data_without_outliers = data[~outliers]

    # Calculate the minimum and maximum of non-outliers
    min_val = np.min(data_without_outliers)
    max_val = np.max(data_without_outliers)

    # Generate n_splits evenly spaced numbers using linspace
    splits = np.linspace(min_val, max_val, n_splits+1)

    # Create an empty array to store the group index of each data point
    group_indices = np.zeros(data.shape, dtype=int)

    # Split non-outliers into n_splits groups
    for i in range(n_splits):
        if i == n_splits - 1:
            group = (data >= splits[i]) & (data <= splits[i+1])
        else:
            group = (data >= splits[i]) & (data < splits[i+1])
        group_indices[group] = i

    # Add outliers back to the first or last group
    for outlier in data_outliers:
        if outlier < min_val:
            group_indices[data == outlier] = 0
        else:
            group_indices[data == outlier] = n_splits - 1

    # Return the group index of each data point
    return group_indices

data = np.array([1, 100, 300, 20, 50, 4, -1])
print(split_data(data, 4))