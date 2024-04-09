import os
from collections import Counter


path = "./homework-2"
# 獲取路徑下的所有內容
all_contents = os.listdir(path)

# 過濾出所有的資料夾
folders = [content for content in all_contents if os.path.isdir(os.path.join(path, content))]

folders =[folder.split("_")[0] for folder in folders]
# print(len(folders), folders)

copy_path = "./result"

# 檢查資料夾是否已存在
if not os.path.exists(copy_path):
    # 如果資料夾不存在，則創建它
    os.mkdir(copy_path)

# 獲取路徑下的所有內容
all_contents = os.listdir(copy_path)

# 過濾出所有的檔案
copy_files = [content for content in all_contents ]
copy_files = [file.split("_")[0] for file in copy_files]
# print(len(copy_files), copy_files)  # 輸出所有的檔案名稱

# 檢查重複繳交
# 計算每個元素的出現次數
counter = Counter(copy_files)

# 過濾出出現次數大於 1 的元素
duplicates = [item for item, count in counter.items() if count > 1]
print("重複繳交:")
print(duplicates)  # 輸出重複的元素

print("--------------------")
# 檢查消失的檔案
disappear_files = [file for file in folders if file not in copy_files]
print("消失的檔案:")
print(disappear_files)
