# Giao diện nhận diện dấu hiệu ngập lụt

Giao diện chính là `index.html`, `css/style.css` và `js/app.js`, được FastAPI phục vụ trực tiếp.

## Chạy

Từ thư mục gốc repository, sau khi cài môi trường và đặt checkpoint:

```powershell
.\.venv\Scripts\python.exe .\flood_detector_server\main.py
```

Mở http://localhost:8000/. Không cần chạy npm cho giao diện này.

## Sử dụng

- **Tệp Video:** chọn video MP4/WebM/OGG, phát/tạm dừng, tua và chỉnh tốc độ.
- **Ảnh Đơn:** tải ảnh JPG/PNG/WebP rồi phân tích.
- **Cài đặt:** đổi WebSocket URL, FPS Limit, ngưỡng confidence và cảnh báo âm thanh.
- **Nhật ký:** chụp snapshot, tải ảnh và xuất CSV.

Chỉ vẽ box khi dự đoán đạt ngưỡng. Điểm confidence cao nhất vẫn hiện trong bảng chỉ số khi chưa có box đạt ngưỡng.

Giao diện hiện tại không có tab Camera, Demo hoặc YouTube. Backend vẫn giữ các WebSocket đã có để sử dụng qua API.

FPS hiển thị là tốc độ xử lý/gửi khung hình, có thể thấp hơn FPS của video. Khi chạy server ở cổng khác, mở giao diện từ chính cổng đó để tự chọn đúng URL API/WebSocket.

Thư mục `src/` và cấu hình Vite là bản giao diện thử nghiệm cũ; hướng dẫn chính dùng trang HTML được FastAPI phục vụ.
