# Serving Model CNN via FastAPI

Một RESTful API được xây dựng bằng FastAPI để phục vụ (serve) các dự đoán từ mô hình Convolutional Neural Network (CNN) đã được huấn luyện trước, sử dụng TensorFlow.

---

## Mục lục

- [Giới thiệu](#giới-thiệu)  
- [Tính năng chính](#tính-năng-chính)  
- [Yêu cầu trước khi cài đặt](#yêu-cầu-trước-khi-cài-đặt)  
- [Cài đặt](#cài-đặt)  
- [Cách chạy ứng dụng (triển khai)](#cách-chạy-ứng-dụng-triển-khai)   

---

## Giới thiệu

Dự án này cung cấp một API đơn giản để nhận ảnh đầu vào (qua HTTP POST) và trả về nhãn dự đoán cùng độ tin cậy (confidence) do mô hình CNN tạo ra.  
Ứng dụng phù hợp để triển khai microservice dự đoán ảnh trong các hệ thống lớn, ví dụ classify mèo – chó, phát hiện vật thể, v.v.

---

## Tính năng chính

- Tải và khởi tạo mô hình CNN TensorFlow đã huấn luyện sẵn (mô hình lenet và vgg16) 
- Nhận ảnh đầu vào qua multipart/form-data  
- Trả về kết quả dự đoán dưới dạng JSON  
- Hỗ trợ auto-reload khi phát triển  

---

## Yêu cầu trước khi cài đặt

- Python 3.8 hoặc mới hơn  
- Git  

---

## Cài đặt

1. **Clone repository về máy**  
   ```bash
   git clone https://github.com/AE-AI-HIT15/AI2025.git
   cd AI2025/HW/khanhnq/baitapapi_copy
2. Cài đặt các thư viện phụ thuộc
   pip install "tensorflow>=2.19.0,<3.0.0" \
            "fastapi>=0.115.12,<0.116.0" \
            "uvicorn>=0.34.2,<0.35.0" \
            "python-multipart>=0.0.20,<0.0.21"
## Cách chạy ứng dụng triển khai
Trong thư mục gốc baitapapi_copy, chạy lệnh:


uvicorn app.main:app --reload --host 0.0.0.0 --port 8090
--reload : tự động tải lại khi code thay đổi (dùng cho phát triển)

Server sẽ lắng nghe tại địa chỉ: http://localhost:8090

