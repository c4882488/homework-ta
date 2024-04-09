step 1. get_file.py
```python
# 修改
# 指定要搜尋的目錄
search_dir = "./homework-2"
```

step 2. identification.py
查看程式碼重複

step 3. check_disappear_file.py (選用)
如果有多或資料夾有但沒有程式碼可以用
```python
# 修改
path = "./homework-2"
```

step 4. iden_2_hw2.py
copy iden_2_hw2.py 更改 iden_2_hw3.py 
```python
# 修改
default_score = 80
nessary_imports = ["train_test_split", "StandardScaler", "LogisticRegression", "plt"]
nessary_programs = ["fit"]
select_comments = ["print(", ".show("]
select_comments_name = ["print_count", "show_count"]
default_select_comments_count = [0, 1]
double_check = ['C110156222', 'C110156240', 'C110156215', 'C109179183']
# ---
# 倍率設定
Similar_magnification = 0.15
simplified_chinese_magnification = 0.15
double_check_magnification = 0.04
# ---
# scaler = MinMaxScaler(feature_range=(0, 5))  # 設定縮放範圍為0到3
# df['平均註解字數_mixmax'] = scaler.fit_transform(df[['平均註解字數']])

df['平均註解字數_mixmax'] = split_data(df, '平均註解字數', 5)

# scaler = MinMaxScaler(feature_range=(0, 4))  # 設定縮放範圍為0到4
# df['代碼行數_mixmax'] = scaler.fit_transform(df[['代碼行數']])

df['代碼行數_mixmax'] = split_data(df, '代碼行數', 5)


# scaler = MinMaxScaler(feature_range=(0, 7))  # 設定縮放範圍為0到4
# df['使用的import數量_mixmax'] = scaler.fit_transform(df[['使用的import數量']])

df['使用的import數量_mixmax'] = split_data(df, '使用的import數量', 7)

# scaler = MinMaxScaler(feature_range=(0, 2))  # 設定縮放範圍為0到3
# df['sum_print_count_mixmax'] = scaler.fit_transform(df[['sum_print_count']])

df['sum_print_count_mixmax'] = split_data(df, 'sum_print_count', 3)

# scaler = MinMaxScaler(feature_range=(0, 6))  # 設定縮放範圍為0到5
# df['sum_show_count_mixmax'] = scaler.fit_transform(df[['sum_show_count']])

df['sum_show_count_mixmax'] = split_data(df, 'sum_show_count', 6)
```

