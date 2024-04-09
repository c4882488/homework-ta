import os
import difflib

def read_file(filename):
    with open(filename, 'r') as file:
        return file.read()

def compare_files(file1, file2):
    content1 = read_file(file1)
    content2 = read_file(file2)

    matcher = difflib.SequenceMatcher(None, content1, content2)
    return matcher.ratio()

def check_similarity(directory, threshold):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.py')]
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            similarity = compare_files(files[i], files[j])
            if similarity > threshold:
                print(f"Files {files[i]} and {files[j]} are {similarity*100}% similar")

# 使用方式
check_similarity('./result', 0.40)