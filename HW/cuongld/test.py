import requests

url = "http://127.0.0.1:8000/predict/"

# Sử dụng dấu gạch chéo xuôi trong đường dẫn
files = {"file": open("C:/Users/admin/Pictures/back ground/f8456800ac55a50acda33ea6b9267e54.jpg", "rb")}
response = requests.post(url, files=files)

print(response.json())

