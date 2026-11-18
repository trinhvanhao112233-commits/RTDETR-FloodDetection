# Chạy nhận diện trực tiếp bằng best.pt

Server dùng Ultralytics RT-DETR và PyTorch để đọc checkpoint, tự chọn GPU CUDA nếu PyTorch nhận GPU; nếu không có GPU thì dùng CPU.

## Khởi động trên máy hiện tại

Mở PowerShell tại thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\flood_detector_server\requirements.txt
.\.venv\Scripts\python.exe .\flood_detector_server\main.py
```

Đặt checkpoint một lớp `flood` (ID 0) tại `flood_detector_server/best.pt`. Không cần export ONNX. Sau khi thay checkpoint, khởi động lại server.

Repository kèm checkpoint amp-3 bằng Git LFS. Khi clone, dùng Git LFS để tải trọng số thực; nếu cần chạy `git lfs pull` tại thư mục gốc repository. File trọng số có dung lượng khoảng 66,2 MB, không phải file con trỏ LFS vài trăm byte.

Mở http://localhost:8000/ để tải ảnh hoặc video lên phân tích. Giao diện hiện tại có hai tab Tệp Video và Ảnh Đơn. Backend vẫn giữ API camera/YouTube; riêng YouTube cần Internet và Node.js, luồng video được đọc vào bộ nhớ mà không lưu toàn bộ xuống đĩa.

Kiểm tra http://localhost:8000/health. Khi dùng GPU, `device` là `cuda:0` và `gpu_name` cho biết tên GPU. Môi trường `.venv` hiện tại đã có PyTorch CUDA. Ở máy khác, nếu `torch.cuda.is_available()` trả về False, cần cài bản PyTorch phù hợp GPU/driver theo https://pytorch.org/get-started/locally/.

## Cấu hình tùy chọn

```powershell
$env:MODEL_PATH = 'D:\models\best.pt'
$env:MODEL_DEVICE = 'cuda:0'
.\.venv\Scripts\python.exe .\flood_detector_server\main.py
```

`MODEL_DEVICE` mặc định là `auto`; có thể đặt `cpu` để kiểm tra trên CPU. Nếu yêu cầu CUDA nhưng không khả dụng, server báo lỗi để tránh âm thầm dùng thiết bị khác.

## Các endpoint

- `POST /api/detect`: ảnh multipart với trường `file`, hoặc raw bytes.
- `WS /ws/flood-detection`: khung hình camera/video.
- `WS /ws/youtube-detection`: nguồn YouTube.
- `GET /health`: checkpoint, runtime và thiết bị đang dùng.

Giao diện chỉ vẽ box và kích hoạt cảnh báo với dự đoán đạt ngưỡng tin cậy đã chọn. Dự đoán dưới ngưỡng không vẽ lên ảnh/video; điểm cao nhất vẫn được hiển thị trong bảng chỉ số.

Các file ONNX cũ vẫn có thể được lưu để tham khảo; server hiện tại chỉ nhận checkpoint `.pt`.
