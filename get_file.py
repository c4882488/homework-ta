import os
import shutil
import zipfile
from rarfile import RarFile
import py7zr

# 指定要搜尋的目錄
search_dir = "./homework-2"

# 指定要複製到的目錄
dest_dir = "./result"

def copy_files(dirpath):
    for dirpath, dirnames, filenames in os.walk(dirpath):
        for filename in filenames:
            # 檢查檔案是否為.py或.ipynb檔案
            if filename.lower().endswith(".py") or filename.lower().endswith(".ipynb"):
                # 獲取檔案的完整路徑
                file_path = os.path.join(dirpath, filename)
                # 複製檔案到目標目錄
                shutil.copy(file_path, dest_dir)

# 遍歷指定目錄及其子目錄下的所有檔案
for dirpath, dirnames, filenames in os.walk(search_dir):
    for filename in filenames:
        # 檢查檔案是否為.zip檔案
        if filename.lower().endswith(".zip"):
            # 獲取檔案的完整路徑
            zip_file_path = os.path.join(dirpath, filename)
            extract_floder = filename.split(".")[0]
            extract_dir = os.path.join(dirpath)

            # 創建目標目錄
            os.makedirs(extract_dir, exist_ok=True)

            # 解壓縮.zip檔案到一個臨時目錄
            temp_dir = os.path.join(dirpath, "temp")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # 將解壓縮的檔案移動到目標目錄
            for filename in os.listdir(temp_dir):
                # 獲取新的檔案名稱
                new_filename = extract_floder
                # 獲取新的檔案路徑
                new_file_path = os.path.join(extract_dir, new_filename)
                # 移動檔案
                shutil.move(os.path.join(temp_dir, filename), new_file_path)

            # 刪除臨時目錄
            shutil.rmtree(temp_dir)

            # 解壓縮檔案都加入學號
            for dirpath, dirnames, filenames in os.walk(os.path.join(extract_dir, new_filename)):
                for filename in filenames:
                    # 獲取檔案的完整路徑
                    file_path = os.path.join(dirpath, filename)
                    # 更改檔案名稱 前綴"aa"
                    new_filename = extract_floder.split("_")[0] + "_"+ filename
                    # 獲取新的檔案路徑
                    new_file_path = os.path.join(dirpath, new_filename)
                    # 更改檔案名稱
                    os.rename(file_path, new_file_path)

            copy_files(dirpath)

        elif filename.lower().endswith(".rar"):
            # 獲取檔案的完整路徑
            rar_file_path = os.path.join(dirpath, filename)
            extract_floder = filename.split(".")[0]
            extract_dir = os.path.join(dirpath)

            # 創建目標目錄
            os.makedirs(extract_dir, exist_ok=True)

            # 解壓縮.rar檔案到一個臨時目錄
            temp_dir = os.path.join(dirpath, "temp")
            with RarFile(rar_file_path, 'r') as rar_ref:
                rar_ref.extractall(temp_dir)

            # 將解壓縮的檔案移動到目標目錄
            for filename in os.listdir(temp_dir):
                # 獲取新的檔案名稱
                new_filename = extract_floder
                # 獲取新的檔案路徑
                new_file_path = os.path.join(extract_dir, new_filename)
                # 移動檔案
                shutil.move(os.path.join(temp_dir, filename), new_file_path)

            # 刪除臨時目錄
            shutil.rmtree(temp_dir)

            # 解壓縮檔案都加入學號
            for dirpath, dirnames, filenames in os.walk(os.path.join(extract_dir, new_filename)):
                for filename in filenames:
                    # 獲取檔案的完整路徑
                    file_path = os.path.join(dirpath, filename)
                    # 更改檔案名稱 前綴"aa"
                    new_filename = extract_floder.split("_")[0] + "_" + filename
                    # 獲取新的檔案路徑
                    new_file_path = os.path.join(dirpath, new_filename)
                    # 更改檔案名稱
                    os.rename(file_path, new_file_path)

            # 解壓縮後再次遍歷該目錄及其子目錄以尋找.py或.ipynb檔案
            copy_files(dirpath)
        elif filename.lower().endswith(".7z"):

            # 獲取檔案的完整路徑
            file_path = os.path.join(dirpath, filename)
            extract_folder = filename.split(".")[0]
            extract_dir = os.path.join(dirpath)

            # 創建目標目錄
            os.makedirs(extract_dir, exist_ok=True)

            # 解壓縮 .7z 檔案到一個臨時目錄
            temp_dir = os.path.join(dirpath, "temp")
            with py7zr.SevenZipFile(file_path, mode='r') as z:
                z.extractall(path=temp_dir)

            # 將解壓縮的檔案移動到目標目錄
            for filename in os.listdir(temp_dir):
                # 獲取新的檔案名稱
                new_filename = extract_folder
                # 獲取新的檔案路徑
                new_file_path = os.path.join(extract_dir, new_filename)
                # 移動檔案
                shutil.move(os.path.join(temp_dir, filename), new_file_path)

            # 刪除臨時目錄
            shutil.rmtree(temp_dir)

            # 解壓縮檔案都加入學號
            for dirpath, dirnames, filenames in os.walk(os.path.join(extract_dir, new_filename)):
                for filename in filenames:
                    # 獲取檔案的完整路徑
                    file_path = os.path.join(dirpath, filename)
                    # 更改檔案名稱 前綴"aa"
                    new_filename = extract_folder.split("_")[0] + "_" + filename
                    # 獲取新的檔案路徑
                    new_file_path = os.path.join(dirpath, new_filename)
                    # 更改檔案名稱
                    os.rename(file_path, new_file_path)

            # 解壓縮後再次遍歷該目錄及其子目錄以尋找.py或.ipynb檔案
            copy_files(dirpath)
        elif filename.lower().endswith(".py") or filename.lower().endswith(".ipynb"):
            # 獲取檔案的完整路徑
            file_path = os.path.join(dirpath, filename)
            # 複製檔案到目標目錄
            shutil.copy(file_path, dest_dir)