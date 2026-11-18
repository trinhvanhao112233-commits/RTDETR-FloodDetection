"""Build a standalone Kaggle notebook for adding labelled images to flood training."""

import json
from pathlib import Path
import textwrap


def cell(kind, source):
    item = {'cell_type': kind, 'metadata': {}, 'source': textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    return item


PREPARE_CODE = r'''
from collections import Counter

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def read_names(cfg):
    names = cfg.get('names', {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): str(v).strip().lower() for k, v in names.items()}

def find_split(root, split, required=False):
    variants = ('valid', 'val') if split == 'val' else (split,)
    candidates = []
    for name in variants:
        for images in (root / name / 'images', root / 'images' / name):
            labels = images.parent / 'labels' if images.name == 'images' else root / 'labels' / name
            if images.is_dir() and labels.is_dir():
                candidates.append((images, labels))
    assert len(candidates) <= 1, f'Nhiều thư mục {split} trong {root}; chỉ giữ một cấu trúc.'
    if required:
        assert candidates, f'Thiếu ảnh/nhãn {split} trong {root}'
    return candidates[0] if candidates else None

def normalize_label(path):
    assert path.is_file(), f'Thiếu nhãn {path}; ảnh không ngập cần file .txt rỗng.'
    output = []
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        assert all(math.isfinite(v) for v in values), f'{path}:{number}: NaN/Infinity'
        assert values[0] == 0, f'{path}:{number}: chỉ cho phép class ID 0'
        coords = values[1:]
        if len(coords) == 4:
            x, y, w, h = coords
        else:
            assert len(coords) >= 6 and len(coords) % 2 == 0, f'{path}:{number}: sai định dạng'
            assert all(0 <= v <= 1 for v in coords), f'{path}:{number}: polygon ngoài ảnh'
            xs, ys = coords[::2], coords[1::2]
            x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1, f'{path}:{number}: box sai'
        assert min(x-w/2, y-h/2) >= -1e-4 and max(x+w/2, y+h/2) <= 1.0001, f'{path}:{number}: box vượt biên'
        output.append(f'0 {x:.9f} {y:.9f} {w:.9f} {h:.9f}')
    # Canonical ordering permits duplicate-image checks when annotation order differs.
    return sorted(output)

def prepare_merged_dataset(old_yaml, new_yaml, destination):
    old_yaml, new_yaml = Path(old_yaml), Path(new_yaml)
    assert old_yaml.resolve() != new_yaml.resolve(), 'Dataset cũ và mới đang trỏ cùng một YAML.'
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    seen, errors, manifest = {}, [], []
    counts, positive = Counter(), Counter()
    added_new_train = 0
    for source_name, source_yaml in (('old', old_yaml), ('new', new_yaml)):
        cfg = yaml.safe_load(source_yaml.read_text(encoding='utf-8-sig'))
        assert read_names(cfg) == {0: 'flood'}, f'{source_yaml}: cần đúng một lớp flood/Flood'
        assert int(cfg.get('nc', 1)) == 1, f'{source_yaml}: nc phải bằng 1'
        for split in ('train', 'val', 'test'):
            folders = find_split(source_yaml.parent, split, required=(split == 'train' or (source_name == 'old' and split == 'val')))
            if folders is None:
                continue
            image_folder, label_folder = folders
            images = sorted(p for p in image_folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            assert images, f'Thư mục ảnh rỗng: {image_folder}'
            for image_path in images:
                try:
                    label_path = label_folder / (image_path.stem + '.txt')
                    lines = normalize_label(label_path)
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        rgb = image.convert('RGB')
                        digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    if digest in seen:
                        previous = seen[digest]
                        assert previous['split'] == split, f'Ảnh trùng giữa {previous["split"]} và {split}: {image_path}'
                        assert previous['labels'] == lines, f'Ảnh trùng nhưng nhãn khác: {image_path}'
                        manifest.append({'source': str(image_path), 'split': split, 'duplicate_skipped': True})
                        continue
                    seen[digest] = {'split': split, 'labels': lines}
                    dst_images, dst_labels = destination / split / 'images', destination / split / 'labels'
                    dst_images.mkdir(parents=True, exist_ok=True)
                    dst_labels.mkdir(parents=True, exist_ok=True)
                    stem = f'{source_name}_{digest}'
                    shutil.copy2(image_path, dst_images / (stem + image_path.suffix.lower()))
                    (dst_labels / (stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
                    counts[f'{source_name}/{split}'] += 1
                    positive[split] += bool(lines)
                    added_new_train += source_name == 'new' and split == 'train'
                    manifest.append({'source': str(image_path), 'split': split, 'boxes': len(lines), 'pixel_sha256': digest})
                except Exception as exc:
                    errors.append(f'{image_path}: {exc}')
    for split in ('train', 'val'):
        if not positive[split]:
            errors.append(f'{split}: cần có ít nhất một ảnh chứa box flood hợp lệ.')
    if not added_new_train:
        errors.append('Không có ảnh train mới hợp lệ được thêm; kiểm tra đường dẫn/ảnh trùng.')
    audit = {'sources': {'old': str(old_yaml), 'new': str(new_yaml)}, 'counts': dict(counts), 'errors': errors, 'files': manifest}
    (destination / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Số ảnh thực sự thêm:', json.dumps(dict(counts), indent=2))
    for error in errors[:20]:
        print(error)
    assert not errors, f'Dừng: {len(errors)} lỗi. Xem {destination / "audit.json"}'
    config = {'path': str(destination.resolve()), 'train': 'train/images', 'val': 'val/images', 'nc': 1, 'names': {0: 'flood'}}
    if (destination / 'test' / 'images').is_dir():
        config['test'] = 'test/images'
    output = destination / 'data_merged.yaml'
    output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return output
'''


def main():
    original = json.loads(Path(__file__).with_name('train_kaggle.ipynb').read_text(encoding='utf-8'))
    callback = next(''.join(c['source']) for c in original['cells'] if ''.join(c['source']).startswith('def stop_on_nonfinite'))
    callback = callback.split('TRAIN_ARGS = dict(')[0]
    cells = [
        cell('markdown', '''
        # Học thêm ảnh đường ngập từ checkpoint cũ

        Upload ba nguồn vào Kaggle Input: checkpoint đang dùng, dataset cũ đầy đủ và dataset ảnh mới đã gán nhãn.
        Bật GPU + Internet. Chạy từng ô từ trên xuống. Notebook không sửa dữ liệu Input.

        Dùng một lớp `0 = flood`. Ảnh ngập có box/polygon; ảnh không ngập có file nhãn rỗng.
        Giữ phân chia train/valid/test sẵn có, không tự chuyển ảnh validation/test sang train.
        Nếu ảnh mới chỉ có train, notebook vẫn dùng validation cũ. Muốn đánh giá tình huống mới, hãy chuẩn bị thêm valid hoặc test theo nguồn/video độc lập.
        Chặn ảnh giống hệt nhau giữa các split; vẫn cần tự tránh khung hình gần nhau của cùng video.

        Cấu hình khởi đầu: 20 epoch, batch 4, AdamW, lr0=1e-5. Đây là một lượt fine-tune mới từ trọng số cũ, resume=False.
        '''),
        cell('code', '%pip install -q "ultralytics==8.4.208"'),
        cell('markdown', '## 1. Liệt kê đường dẫn Input'),
        cell('code', '''
        from pathlib import Path
        import hashlib, json, math, shutil, gc
        from datetime import datetime
        import yaml, torch
        from PIL import Image
        from ultralytics import RTDETR

        print('Các file YAML:')
        for p in sorted(Path('/kaggle/input').rglob('*.yaml')):
            print(p)
        print('Các checkpoint:')
        for p in sorted(Path('/kaggle/input').rglob('*.pt')):
            print(p)
        assert torch.cuda.is_available(), 'Bật GPU trong Kaggle Settings.'
        print('GPU:', torch.cuda.get_device_name(0))
        '''),
        cell('markdown', '''
        ## 2. Điền đúng ba đường dẫn vừa in
        BASE_WEIGHTS là file best.pt đã sao chép từ server hiện tại, không chọn bản epoch 40 trong Downloads.
        OLD_DATASET_YAML là data.yaml của bộ cũ; NEW_DATASET_YAML là data.yaml của bộ hơn 50 ảnh mới.
        Cả hai hỗ trợ train/images + valid/images (hoặc val/images) và labels tương ứng; cũng hỗ trợ images/train + labels/train.
        Ảnh mới có thể chỉ có train/images + train/labels. Đừng thêm ảnh thô chưa gán nhãn.
        '''),
        cell('code', '''
        BASE_WEIGHTS = None       # Điền chuỗi đường dẫn .pt trong Input
        OLD_DATASET_YAML = None    # Điền chuỗi đường dẫn data.yaml bộ cũ
        NEW_DATASET_YAML = None    # Điền chuỗi đường dẫn data.yaml bộ mới
        EPOCHS = 20
        BATCH = 4                 # Giảm còn 2 nếu thiếu VRAM
        LEARNING_RATE = 1e-5
        # SHA256 của checkpoint đang dùng trên server, giúp tránh chọn nhầm phiên bản.
        EXPECTED_BASE_SHA256 = '30ddbadd0c75bd7279916089b90b934ae9b09136df8a047b23cf1496cc94fd44'

        for label, value in [('BASE_WEIGHTS', BASE_WEIGHTS), ('OLD_DATASET_YAML', OLD_DATASET_YAML), ('NEW_DATASET_YAML', NEW_DATASET_YAML)]:
            assert value and Path(value).is_file(), f'Điền đúng {label} ở ô này.'
        BASE_SHA256 = hashlib.sha256(Path(BASE_WEIGHTS).read_bytes()).hexdigest()
        print('Checkpoint SHA256:', BASE_SHA256)
        assert BASE_SHA256 == EXPECTED_BASE_SHA256, 'Checkpoint khác bản server hiện tại. Kiểm tra lại file đã upload; chỉ đổi EXPECTED_BASE_SHA256 khi chủ động chọn mô hình khác.'
        WORK = Path('/kaggle/working')
        SESSION_DIR = WORK / ('finetune_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        SESSION_DIR.mkdir(parents=True, exist_ok=False)
        '''),
        cell('markdown', '## 3. Kiểm tra và gộp bộ cũ với bộ mới'),
        cell('code', PREPARE_CODE),
        cell('code', '''
        DATA_YAML = prepare_merged_dataset(OLD_DATASET_YAML, NEW_DATASET_YAML, SESSION_DIR / 'dataset')
        (SESSION_DIR / 'training_manifest.json').write_text(json.dumps({
            'base_checkpoint': str(BASE_WEIGHTS), 'base_sha256': BASE_SHA256,
            'old_yaml': str(OLD_DATASET_YAML), 'new_yaml': str(NEW_DATASET_YAML),
            'merged_yaml': str(DATA_YAML), 'epochs': EPOCHS, 'batch': BATCH, 'lr0': LEARNING_RATE,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Dataset đã gộp:', DATA_YAML)
        '''),
        cell('markdown', '## 4. Đo mô hình cũ trên validation đã gộp để có mốc so sánh'),
        cell('code', '''
        baseline_model = RTDETR(str(BASE_WEIGHTS))
        assert {int(k): str(v).strip().lower() for k, v in baseline_model.names.items()} == {0: 'flood'}, 'Checkpoint phải có một lớp flood.'
        baseline_metrics = baseline_model.val(data=str(DATA_YAML), split='val', device=0,
            imgsz=640, batch=BATCH, project=str(SESSION_DIR), name='baseline_val', plots=True)
        BASELINE = {k: float(v) for k, v in baseline_metrics.results_dict.items()}
        del baseline_model
        gc.collect()
        torch.cuda.empty_cache()
        print('Validation mô hình cũ:', BASELINE)
        '''),
        cell('markdown', '## 5. Fine-tune từ checkpoint cũ — không khởi tạo lại từ rtdetr-l.pt'),
        cell('code', callback + '''
model = RTDETR(str(BASE_WEIGHTS))
model.add_callback('on_fit_epoch_end', stop_on_nonfinite)
model.train(
    data=str(DATA_YAML), epochs=EPOCHS, imgsz=640, batch=BATCH, device=0, workers=2,
    optimizer='AdamW', lr0=LEARNING_RATE, warmup_epochs=1, warmup_bias_lr=0.0,
    mosaic=0.0, mixup=0.0, amp=True, deterministic=False, seed=0,
    resume=False, patience=10, save=True, plots=True,
    project=str(SESSION_DIR), name='training', exist_ok=False,
)
RUN_DIR = Path(model.trainer.save_dir)
del model
gc.collect()
torch.cuda.empty_cache()
print('Kết quả:', RUN_DIR)
'''),
        cell('markdown', '''
        ## 6. So sánh trên cùng validation và chuẩn bị tải xuống
        Chọn best.pt theo kết quả validation của lượt mới. Không tự thay model trên server.
        mAP chỉ đánh giá box; cần thử cả ảnh/video đường ngập và không ngập độc lập để kiểm tra báo nhầm/bỏ sót.
        '''),
        cell('code', '''
        fine_model = RTDETR(str(RUN_DIR / 'weights' / 'best.pt'))
        fine_metrics = fine_model.val(data=str(DATA_YAML), split='val', device=0,
            imgsz=640, batch=BATCH, project=str(SESSION_DIR), name='finetuned_val', plots=True)
        comparison = {'baseline': BASELINE, 'finetuned': {k: float(v) for k, v in fine_metrics.results_dict.items()}}
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        FINAL_WEIGHTS = WORK / 'best_finetuned.pt'
        shutil.copy2(RUN_DIR / 'weights' / 'best.pt', FINAL_WEIGHTS)
        shutil.copy2(RUN_DIR / 'results.csv', WORK / 'finetune_results.csv')
        (WORK / 'finetune_comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
        print('Tải về:', FINAL_WEIGHTS)
        print('Tải thêm: /kaggle/working/finetune_results.csv và finetune_comparison.json')
        print('Dùng Save Version để lưu notebook và output sau khi hoàn tất.')
        '''),
    ]
    notebook = {'cells': cells, 'metadata': original['metadata'], 'nbformat': 4, 'nbformat_minor': 2}
    target = Path(__file__).with_name('finetune_kaggle.ipynb')
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Created {target}: {len(cells)} cells')


if __name__ == '__main__':
    main()
