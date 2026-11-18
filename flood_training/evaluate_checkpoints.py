"""Run flood checkpoints on a folder of images; no ground-truth metrics implied."""

import argparse
import csv
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conf', type=float, default=0.45)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(args.output / 'ultralytics_config'))

    import torch
    from PIL import Image, ImageDraw
    from ultralytics import RTDETR

    images = sorted(p for p in args.source.iterdir()
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'})
    if not images:
        raise ValueError('No source images found')
    report = {'confidence_threshold': args.conf, 'images': len(images),
              'note': 'No ground-truth labels: detection counts are not accuracy.',
              'models': {}}
    rows = []
    for weights in args.weights:
        model = RTDETR(str(weights))
        print(f'Checkpoint: {weights.name}, names={model.names}, '
              f'epoch={model.ckpt.get("epoch")}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}',
              flush=True)
        out = args.output / weights.stem
        out.mkdir()
        sheet = Image.new('RGB', (1600, ((len(images) + 4) // 5) * 220), 'white')
        draw = ImageDraw.Draw(sheet)
        detected = 0
        for index, path in enumerate(images, 1):
            # Low threshold retains scores for comparison of multiple thresholds.
            result = model.predict(source=str(path), conf=0.001, imgsz=640,
                                   device=args.device, verbose=False)[0]
            boxes = result.boxes
            max_score = float(boxes.conf.max()) if boxes is not None and len(boxes) else 0.0
            selected = result[boxes.conf >= args.conf] if boxes is not None else result
            count = len(selected.boxes) if selected.boxes is not None else 0
            detected += count > 0
            filename = f'{index:03d}_{path.stem}.jpg'
            annotated = Image.fromarray(selected.plot()[:, :, ::-1])
            annotated.save(out / filename, quality=90)
            annotated.thumbnail((312, 184))
            x, y = ((index - 1) % 5) * 320, ((index - 1) // 5) * 220
            sheet.paste(annotated, (x, y))
            draw.text((x + 4, y + 186), f'{index:03d} | {"FLOOD" if count else "NO DETECTION"} | {max_score:.3f}', fill='black')
            draw.text((x + 4, y + 202), path.name[:42].encode('ascii', 'replace').decode(), fill='black')
            rows.append({'model': weights.stem, 'image_id': index,
                         'source': str(path.resolve()), 'max_confidence': round(max_score, 6),
                         'boxes_at_threshold': count, 'flood_detected': bool(count),
                         'message': 'Có dấu hiệu ngập lụt' if count else 'Chưa phát hiện dấu hiệu ngập lụt',
                         'annotated_image': str((out / filename).resolve())})
            print(f'{weights.stem} {index}/{len(images)}: {path.name}: max={max_score:.3f}, boxes={count}', flush=True)
        sheet.save(out / 'contact_sheet.jpg', quality=90)
        report['models'][weights.stem] = {
            'checkpoint': str(weights.resolve()), 'epoch_metadata': model.ckpt.get('epoch'),
            'names': model.names, 'detected_images': detected,
            'undetected_images': len(images) - detected,
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with (args.output / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
