# Huấn luyện RT-DETR một lớp flood

## Notebook Kaggle

- `train_kaggle.ipynb`: huấn luyện từ pretrained `rtdetr-l.pt`, mặc định 80 epoch.
- `finetune_kaggle.ipynb`: học thêm từ checkpoint cũ, mặc định 20 epoch, batch 4, AdamW và lr0=1e-5.
- `build_finetune_notebook.py`: tạo lại notebook fine-tune dựa trên notebook huấn luyện.

Notebook dùng một lớp `0 = flood`, chấp nhận tên `flood` hoặc `Flood`. Bật GPU và Internet trên Kaggle, import notebook và chạy từng ô từ trên xuống.

## Dữ liệu

Ví dụ cấu trúc Roboflow:

```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/             # Tùy chọn, giữ riêng để đánh giá cuối
```

```yaml
train: train/images
val: valid/images
nc: 1
names: ['flood']
```

Nhãn YOLO detection: `0 x_center y_center width height`, chuẩn hóa tọa độ về [0, 1]. Ảnh không ngập cần có file nhãn rỗng. Notebook cũng hỗ trợ chuyển polygon thành bounding box.

## Học thêm ảnh mới

Thêm ba nguồn vào Kaggle Input: checkpoint cũ, dataset cũ đầy đủ và dataset ảnh mới đã gán nhãn. Trong `finetune_kaggle.ipynb`, điền:

```python
BASE_WEIGHTS = '/kaggle/input/.../best.pt'
OLD_DATASET_YAML = '/kaggle/input/.../data.yaml'
NEW_DATASET_YAML = '/kaggle/input/.../data.yaml'
```

Kiểm tra `EXPECTED_BASE_SHA256`: đây là fingerprint của checkpoint chọn làm mốc khi notebook được tạo. Khi chủ động chọn checkpoint mới, thay bằng SHA256 của file đó. Trên Windows có thể lấy hash bằng:

```powershell
Get-FileHash -LiteralPath .\flood_detector_server\best.pt -Algorithm SHA256
```

Notebook giữ split sẵn có, gộp dữ liệu, chặn ảnh giống hệt giữa các split và so sánh checkpoint cũ/mới trên cùng validation. Vẫn cần tránh các khung hình gần nhau của cùng video nằm ở nhiều split.

`resume=False` trong notebook tạo lượt fine-tune mới từ checkpoint đã nạp. Cấu hình 20 epoch/lr0=1e-5 là điểm khởi đầu để thử, cần đánh giá lại trước khi triển khai.

Sau khi train, lưu version và tải `best_finetuned.pt`, `finetune_results.csv`, `finetune_comparison.json`. Giữ model cũ để so sánh trên bộ test đường phố độc lập.

## Công cụ khác

- `evaluate_checkpoints.py`: chạy checkpoint trên thư mục ảnh và lưu ảnh nhận diện/CSV; số ảnh có dự đoán không phải accuracy.
- `train.py`: script local cũ, yêu cầu tự chuẩn bị `flood_dataset/data.yaml` và dữ liệu tương ứng. Dataset cục bộ không được đưa lên Git.
- `export.py`: công cụ xuất ONNX cũ, không cần cho server đang dùng trực tiếp `.pt`.
- `merge_class.py`: công cụ cũ sửa nhãn tại chỗ bằng cách gộp mọi class về 0. Không chạy với dataset một lớp hiện tại hoặc khi các class khác không có nghĩa là ngập.

Checkpoint `best.pt` tốt nhất của mỗi lượt được chọn theo metric validation; hậu tố thư mục như `amp-3` chỉ là số thứ tự lượt chạy.
