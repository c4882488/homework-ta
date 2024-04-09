import os
import numpy as np
import ssdeep
import pandas as pd
import tokenize
import io
import re
from sklearn.preprocessing import MinMaxScaler
from opencc import OpenCC

def calculate_fuzzy_hash(file_path):
    with open(file_path, "rb") as f:
        file_data = f.read()
        fuzzy_hash = ssdeep.hash(file_data)
    return fuzzy_hash


def compare_similarity(file1, file2):
    hash1 = calculate_fuzzy_hash(file1)
    hash2 = calculate_fuzzy_hash(file2)

    similarity = ssdeep.compare(hash1, hash2)
    return similarity


def find_comments(filename):
    with open(filename, 'r') as file:
        source_code = file.read()

    comments = []

    # 使用tokenize模組來解析Python源碼
    tokens = tokenize.generate_tokens(io.StringIO(source_code).readline)

    for token in tokens:
        if token.type == tokenize.COMMENT:
            comment = token.string.lstrip("# ''' \"\"\" ")
            # 移除所有非中文、非英文的字符
            comment = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', comment)

            # 分割註解並計算單詞的數量
            eng_comment = re.sub(r'[^a-zA-Z]', ' ', comment)
            eng_comment = eng_comment.strip()
            word_count = len(eng_comment.split())
            if len(eng_comment) == 0:
                word_count = 0

            che_comment = re.sub(r'[^\u4e00-\u9fa5]', '', comment)
            che_comment = che_comment.strip()
            word_count += len(che_comment)
            comments.append((token.start[0], comment,word_count))

    summer_data = {
        "total_rows": len(comments),
        "total_words": sum([word_count for _,_,word_count in comments])
    }

    return summer_data


def find_imports(filename):
    with open(filename, 'r') as file:
        source_code = file.read()

    imports = []

    # 使用tokenize模組來解析Python源碼
    tokens = tokenize.generate_tokens(io.StringIO(source_code).readline)

    for token in tokens:
        if token.type == tokenize.NAME and token.string == "import":
            # 獲取下一個 token，這應該是 import 語句的一部分
            # imports.append(next(tokens).string)
            next_token = next(tokens)
            imports.append(next_token.string)
            while next_token.string != "as" and next_token.string != "\n":
                next_token = next(tokens)
            # 如果遇到 "as"，則獲取 "as" 之後的 token
            if next_token.string == "as":
                as_value = next(tokens).string
                imports.append(as_value)
        elif token.type == tokenize.NAME and token.string == "from":
            # 獲取下一個 token，這應該是 import 語句的一部分
            next(tokens)
            # 直接找尋 import
            while next(tokens).string != "import":
                next(tokens)

            imports.append(next(tokens).string)
            # 持續尋找直到這行結束
            while next(tokens).string != "\n":
                imports.append(next(tokens).string)

    # 移除重複的import
    imports = list(set(imports))
    # 移除未使用套件
    used_imports = check_imports_usage(filename, imports)

    return used_imports


def check_imports_usage(filename, items):
    with open(filename, 'r') as file:
        source_code = file.read()

    # 使用tokenize模組來解析Python源碼
    tokens = tokenize.generate_tokens(io.StringIO(source_code).readline)

    # 建立一個集合來存儲使用過的項目
    used_items = []

    skip_line = False
    for token in tokens:
        if token.string == "import" or token.string == "from":
            skip_line = True
        elif token.string == "\n":
            skip_line = False

        if skip_line:
            continue

        if token.type == tokenize.NAME and token.string in items:
            used_items.append(token.string)

    # 移除重複的import
    used_items = list(set(used_items))

    return used_items


def check_nessary_imports(use_imports, nessary_imports):
    not_used_imports = []
    for nessary_import in nessary_imports:
        if nessary_import not in use_imports:
            not_used_imports.append(nessary_import)

    nessary_data = {
        "not_used_imports": not_used_imports,
        "length": len(not_used_imports)
    }
    return nessary_data

def check_nessary_programs(filename, nessary_programs):
    not_used_programs = []
    with open(filename, 'r') as file:
        source_code = file.read()

    for nessary_program in nessary_programs:
        if nessary_program not in source_code:
            not_used_programs.append(nessary_program)

    nessary_data = {
        "not_used_programs": not_used_programs,
        "length": len(not_used_programs)
    }

    return nessary_data


def check_comments_count(filename, select_comments):
    with open(filename, 'r') as file:
        source_code = file.read()

    comments = [
        {
            "comment": select_comment,
            "length": source_code.count(select_comment)
        } for select_comment in select_comments
    ]

    return comments


def extract_chinese(text):
    chinese_pattern = re.compile('[\u4e00-\u9fa5]')  # 匹配中文字符的正則表達式
    chinese_chars = chinese_pattern.findall(text)
    chinese_text = ''.join(chinese_chars)
    return chinese_text


def contains_only_simplified_chinese_in_file(filename):
    cc = OpenCC('t2s')  # 繁體轉簡體
    with open(filename, 'r', encoding='utf-8') as file:
        source_code = file.read()

    chinese_text = extract_chinese(source_code)

    simplified_text = cc.convert(chinese_text)

    if len(chinese_text) == 0:
        return 0
    if simplified_text != chinese_text:
        return 0
    return 1

def count_code_lines(filename):
    with open(filename, 'r') as file:
        source_code = file.read()

    # 使用tokenize模組來解析Python源碼
    tokens = tokenize.generate_tokens(io.StringIO(source_code).readline)

    line_count = 0
    for token in tokens:
        # 忽略註解、import、from和只包含空白字符的行
        if token.type != tokenize.COMMENT and token.string != "import" and token.string != "from" and token.line.strip() != '':
            # 只計算一次每一行
            if token.start[0] > line_count:
                line_count = token.start[0]

    return line_count

def split_data(df, column_name, n_splits):
    data = df[column_name].values

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


folder_path = "./hw1_result"
files = [f for f in os.listdir(folder_path) if f.endswith(".py")]
# files = files[:3]
data = []
default_score = 80
nessary_imports = ["train_test_split", "StandardScaler"]
nessary_programs = ["describe"]
select_comments = ["print(", ".show("]
select_comments_name = ["print_count", "show_count"]
default_select_comments_count = [5, 0]

# ---
# 倍率設定
Similar_magnification = 0.15
simplified_chinese_magnification = 0.1

for i in range(len(files)):

    file1 = os.path.join(folder_path, files[i])

    Similar_work = []
    Similar_score = []

    # 排除自己
    files_2 = files.copy()
    files_2.pop(i)
    for file_2 in files_2:
        file2 = os.path.join(folder_path, file_2)

        similarity = compare_similarity(file1, file2)
        if similarity > 39:
            # print(f"比較 {files[i]} 和 {files[j]} 的相似度：{similarity}")
            Similar_work.append(file_2)
            Similar_score.append(similarity)

    if len(Similar_score) == 0:
        Similar_score.append(0)

    comments_data = find_comments(file1)
    # print(comments_data)

    import_data = find_imports(file1)
    # print(import_data)

    nessary_data = check_nessary_imports(import_data, nessary_imports)
    # print(nessary_data)

    nessary_programs_data = check_nessary_programs(file1, nessary_programs)
    # print(nessary_programs_data)

    comments_counts = check_comments_count(file1, select_comments)
    # print(comments_count)

    simplified_chinese = contains_only_simplified_chinese_in_file(file1)
    # print(simplified_chinese)
    # print(file1)

    code_lines = count_code_lines(file1)

    # list to string
    Similar_works = ', '.join(Similar_work)
    import_datas = ', '.join(import_data)
    not_used_imports = ', '.join(nessary_data["not_used_imports"])

    comments = []

    for idx, comments_count in enumerate(comments_counts):
        comments.extend([comments_count["length"], default_select_comments_count[idx], comments_count["length"] - default_select_comments_count[idx]])

    # 平均註解字數
    if comments_data["total_words"] != 0:
        avg_comments_words = comments_data["total_words"] / comments_data["total_rows"]
    else:
        avg_comments_words = 0
    # print([
    #         files[i],  # 作業檔案名稱
    #         Similar_works,  # 與誰相似
    #         len(Similar_work),  # 相似次數
    #         np.mean(Similar_score),  #平均相似度
    #         np.mean(Similar_score)/100,  # 相似基數
    #         Similar_magnification,  # 相似倍率
    #         simplified_chinese,  # 是否為簡字基數
    #         simplified_chinese_magnification,  # 簡字倍率
    #         comments_data["total_rows"],  # 註解總行數
    #         comments_data["total_words"],  # 註解總字數
    #         avg_comments_words,  # 平均註解字數
    #         import_datas,  # 使用的import
    #         not_used_imports,  # 未使用的import
    #         len(import_data) - nessary_data["length"],  # 使用的import數量
    #         code_lines  # 代碼行數
    #     ])
    data.append(
        [
            files[i],  # 作業檔案名稱
            Similar_works,  # 與誰相似
            len(Similar_work),  # 相似次數
            np.mean(Similar_score),  #平均相似度
            np.mean(Similar_score)/100,  # 相似基數
            Similar_magnification,  # 相似倍率
            simplified_chinese,  # 是否為簡字基數
            simplified_chinese_magnification,  # 簡字倍率
            comments_data["total_rows"],  # 註解總行數
            comments_data["total_words"],  # 註解總字數
            avg_comments_words,  # 平均註解字數
            import_datas,  # 使用的import
            not_used_imports,  # 未使用的import
            len(import_data) - nessary_data["length"],  # 使用的import數量
            code_lines  # 代碼行數
        ] + comments
    )

comments = []

for idx, select_comment_name in enumerate(select_comments_name):
    comments.extend([select_comment_name, f"default_{select_comment_name}", f"sum_{select_comment_name}"])

df = pd.DataFrame(data, columns=["作業檔案名稱", "與誰相似", "相似次數", "平均相似度", "相似基數", "相似倍率", "是否為簡字基數", "簡字倍率", "註解總行數", "註解總字數", "平均註解字數", "使用的import", "未使用的import", "使用的import數量","代碼行數"]+comments)

# 假設你的DataFrame是df，你想要縮放的列是'column_name'

df['平均註解字數_mixmax'] = split_data(df, '平均註解字數', 7)

df['代碼行數_mixmax'] = split_data(df, '代碼行數', 3)

df['使用的import數量_mixmax'] = split_data(df, '使用的import數量', 6)

df['sum_print_count_mixmax'] = split_data(df, 'sum_print_count', 3)

scaler = MinMaxScaler(feature_range=(0, 5))  # 設定縮放範圍為0到5
df['sum_show_count_mixmax'] = scaler.fit_transform(df[['sum_show_count']])
# df['sum_show_count_mixmax'] = split_data(df, 'sum_show_count', 6)

# 計算分數
for index, row in df.iterrows():
    score = default_score
    score = score*(1-row["相似基數"] * row["相似倍率"])
    score = score*(1-row["是否為簡字基數"] * row["簡字倍率"])

    score += row["平均註解字數_mixmax"]
    score += row["代碼行數_mixmax"]
    score += row["使用的import數量_mixmax"]
    score += row["sum_print_count_mixmax"]
    score += row["sum_show_count_mixmax"]
    # 分數四捨五入到整數
    score = round(score)
    df.loc[index, "分數"] = score

df.sort_values(by=['作業檔案名稱'], inplace=True)

df.to_excel("comparison_results_hw1.xlsx", index=False)

df2 = df[["作業檔案名稱", "分數"]]

df2.to_csv("result_hw1.csv", index=False)