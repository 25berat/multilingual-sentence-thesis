import os

path = r"C:\Users\berat\PycharmProjects\PaSeMiLL\data\train"

txt_files = [f for f in os.listdir(path) if f.endswith(".txt")]

print("Number of .txt files:", len(txt_files))