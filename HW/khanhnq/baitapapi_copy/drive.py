#!/usr/bin/env python3
import os
import gdown

# URL của folder trên Google Drive
FOLDER_URL = 'https://drive.google.com/drive/folders/1AS5oTybX4fGji3e-89n6WbtFY8a8umsy'
# Thư mục đích muốn lưu
OUTPUT_DIR = 'D:\README\demo'

def download_folder_if_needed(url: str, output: str):
    if not os.path.isdir(output):
        print(f"Folder '{output}' chưa tồn tại, bắt đầu tải về…")
        # download_folder có sẵn trong gdown
        gdown.download_folder(url=url, output=output, quiet=False, use_cookies=False)
        print("Hoàn tất tải folder.")
    else:
        print(f"Folder '{output}' đã có sẵn, bỏ qua download.")

if __name__ == '__main__':
    # Nếu bạn chưa cài gdown: pip install gdown
    download_folder_if_needed(FOLDER_URL, OUTPUT_DIR)
